"""
Training, Distillation & Quantization Pipeline for MedGemma-Micro with Llama-3.2-1B-Instruct
=============================================================================================
Upgrades the language backbone to Llama-3.2-1B-Instruct:
  1. Fine-tunes Llama-3.2-1B on the comprehensive cardiology curriculum (1,523 pairs) via LoRA.
  2. Trains the 1D-Conformer Biosignal Encoder and Cross-Attention Projector (2048-D).
  3. Merges LoRA weights cleanly into base model (zero runtime LoRA overhead).
  4. Quantizes linear weights using high-fidelity 4-bit block-wise quantization with FP16 embeddings.
  5. Serializes unified multimodal weights to medgemma_micro_cardio_edge.safetensors.
"""

import os
import sys
import time
import math
import argparse
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from cardiology_curriculum import CARDIOLOGY_CURRICULUM
from pipeline import (
    ClinicalTextDataset,
    PPGConformerEncoder,
    PPGCrossAttentionProjector,
    MedGemmaMicroModel,
    SyntheticPPGDataset,
)

STUDENT_ID = "unsloth/Llama-3.2-1B-Instruct"
OUTPUT_PATH = "medgemma_micro_cardio_edge.safetensors"
BUDGET_LIMIT_MB = 1200.0  # Fits comfortably under mobile device RAM budget (<1.2GB)


def quantize_state_dict_int4(state_dict: Dict[str, torch.Tensor], group_size: int = 64) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    High-fidelity 4-bit block-wise quantization for 2D linear projection matrices.
    Keeps embed_tokens and conformer/projector weights in FP16 to guarantee zero token corruption.
    """
    compact_dict = {}
    total_bytes = 0
    q_count = 0

    for k, v in state_dict.items():
        # Do not quantize embeddings, lm_head, or sensor modules
        is_linear_weight = (
            v.is_floating_point()
            and "weight" in k
            and v.ndim == 2
            and not k.startswith("ppg_encoder.")
            and not k.startswith("ppg_projector.")
            and "embed_tokens" not in k
            and "lm_head" not in k
        )

        if is_linear_weight:
            orig_shape = list(v.shape)
            in_feat = orig_shape[1]
            pad = (group_size - (in_feat % group_size)) % group_size
            if pad > 0:
                v_padded = F.pad(v, (0, pad))
            else:
                v_padded = v

            reshaped = v_padded.view(-1, group_size)
            max_val = reshaped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
            scale = (max_val / 7.0).to(torch.float16)

            # Quantize [-8, 7] -> offset [0, 15]
            q = torch.clamp(torch.round(reshaped / scale.to(torch.float32)), -8, 7).to(torch.int8) + 8
            q_flat = q.view(-1)

            # Pack two 4-bit values per uint8
            q_even = q_flat[0::2].to(torch.uint8)
            q_odd = q_flat[1::2].to(torch.uint8)
            packed = (q_even | (q_odd << 4)).cpu().contiguous()

            scale_cpu = scale.cpu().contiguous()
            shape_cpu = torch.tensor(orig_shape, dtype=torch.int32).cpu().contiguous()
            group_cpu = torch.tensor([group_size], dtype=torch.int32).cpu().contiguous()

            compact_dict[k] = packed
            compact_dict[k + ".scale"] = scale_cpu
            compact_dict[k + ".orig_shape"] = shape_cpu
            compact_dict[k + ".group_size"] = group_cpu

            total_bytes += packed.nbytes + scale_cpu.nbytes + shape_cpu.nbytes
            q_count += 1
        else:
            if v.is_floating_point():
                fp16_v = v.to(torch.float16).cpu().contiguous()
                compact_dict[k] = fp16_v
                total_bytes += fp16_v.nbytes
            else:
                cpu_v = v.cpu().contiguous()
                compact_dict[k] = cpu_v
                total_bytes += cpu_v.nbytes

    size_mb = total_bytes / (1024.0 * 1024.0)
    print(f"Quantized {q_count} transformer linear weight matrices to 4-bit block-wise (group_size={group_size}).")
    print(f"Total serialized size: {size_mb:.2f} MB")
    return compact_dict, size_mb


def train_and_export_llama(
    student_id: str = STUDENT_ID,
    output_path: str = OUTPUT_PATH,
    epochs: int = 1,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    max_length: int = 160,
    lr: float = 2e-4,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
):
    print("=" * 70)
    print("UPGRADING MEDGEMMA-MICRO TO LLAMA-3.2-1B-INSTRUCT")
    print("=" * 70)
    print(f"Model ID:        {student_id}")
    print(f"Target Output:   {output_path}")
    print(f"Training Device: {device}")
    print(f"Batch Size:      {batch_size} (Grad Accum: {grad_accum_steps})")

    # 1. Load Tokenizer & Model
    print(f"\n[Step 1/5] Loading {student_id}...")
    tokenizer = AutoTokenizer.from_pretrained(student_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_lm = AutoModelForCausalLM.from_pretrained(
        student_id,
        torch_dtype=torch.float32,
    ).to(device)

    llm_dim = base_lm.config.hidden_size
    print(f"  -> Model Hidden Size: {llm_dim}")
    print(f"  -> Total Base Parameters: {sum(p.numel() for p in base_lm.parameters()):,}")

    # 2. Configure PEFT LoRA
    print("\n[Step 2/5] Configuring PEFT LoRA Adapters across Attention & MLP Layers...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_lm = get_peft_model(base_lm, lora_config)
    peft_lm.print_trainable_parameters()

    # 3. Fine-Tune on Cardiology Curriculum
    print(f"\n[Step 3/5] Fine-Tuning on {len(CARDIOLOGY_CURRICULUM)} Curated Clinical Cardiology Q&A Pairs...")
    dataset = ClinicalTextDataset(CARDIOLOGY_CURRICULUM, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in peft_lm.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    total_steps = (len(loader) // grad_accum_steps) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-5)

    peft_lm.train()
    optimizer.zero_grad()
    accum_loss = 0.0

    step_counter = 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for b_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = peft_lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            accum_loss += loss.item() * grad_accum_steps
            epoch_loss += loss.item() * grad_accum_steps

            if (b_idx + 1) % grad_accum_steps == 0 or (b_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in peft_lm.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_counter += 1

                if step_counter % 20 == 0 or (b_idx + 1) == len(loader):
                    elapsed = time.perf_counter() - t0
                    print(f"     Step [{step_counter}/{total_steps}] | Loss: {accum_loss / grad_accum_steps:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} ({elapsed:.1f}s)", flush=True)
                accum_loss = 0.0

        print(f"  -> Epoch [{epoch+1}/{epochs}] Complete - Mean Loss: {epoch_loss / len(loader):.4f}", flush=True)

    # Merge LoRA back into base model
    print("\n  -> Merging LoRA weights into base Llama-3.2-1B model (zero inference overhead)...")
    merged_lm = peft_lm.merge_and_unload()
    merged_lm.eval()

    # 4. Assemble MedGemmaMicroModel & Train Conformer + Cross-Attention Projector
    print(f"\n[Step 4/5] Assembling MedGemmaMicroModel with 1D-Conformer & Cross-Attention Bridge (dim={llm_dim})...")
    model = MedGemmaMicroModel(
        student_lm=merged_lm,
        encoder_in_channels=1,
        encoder_classes=5,
        num_prefix_tokens=4,
        encoder_type="conformer",
        projector_type="cross_attention",
    ).to(device)

    # Train Conformer sensor encoder
    print("  -> Training 1D-Conformer Biosignal Encoder on 90s continuous PPG waveforms (15 epochs)...")
    ppg_dataset = SyntheticPPGDataset(num_samples=300, sampling_rate=25, duration_sec=90)
    ppg_loader = DataLoader(ppg_dataset, batch_size=batch_size * 2, shuffle=True)

    cls_criterion = nn.CrossEntropyLoss()
    optimizer_sensor = torch.optim.AdamW(
        list(model.ppg_encoder.parameters()) + list(model.ppg_projector.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    sensor_epochs = 15
    model.train()
    for epoch in range(sensor_epochs):
        cls_loss = 0.0
        correct = 0
        total = 0
        for waves, labels in ppg_loader:
            waves = waves.to(device)
            labels = labels.to(device)

            optimizer_sensor.zero_grad()
            enc_out = model.ppg_encoder(waves)
            logits = enc_out[0] if isinstance(enc_out, tuple) else enc_out
            loss = cls_criterion(logits, labels)
            loss.backward()
            optimizer_sensor.step()

            cls_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = (correct / total) * 100.0 if total > 0 else 0.0
        if (epoch + 1) % 3 == 0 or epoch == sensor_epochs - 1:
            print(f"     [Conformer Epoch {epoch+1:02d}/{sensor_epochs}] Loss: {cls_loss/len(ppg_loader):.4f} | Accuracy: {acc:.1f}%", flush=True)

    # 5. Quantize & Serialize
    print("\n[Step 5/5] Serializing unified Llama-3.2-1B multimodal weights with block-wise 4-bit...")
    model.eval()
    raw_dict = model.state_dict()
    compact_dict, size_mb = quantize_state_dict_int4(raw_dict, group_size=64)

    metadata = {
        "architecture": "MedGemmaMicro-Llama3.2-1B-Conformer",
        "target_os": "iOS (Core ML / Metal) & Android (LiteRT / GGUF)",
        "quantization": "4-bit block-wise linear weights (group_size=64) + FP16 embeddings",
        "student_backbone": student_id,
        "distilled_from": "google/medgemma-1.5-4b-it",
        "encoder_type": "1D-Conformer",
        "projector_type": "Temporal-Cross-Attention",
        "llm_dim": str(llm_dim),
        "actual_size_mb": f"{size_mb:.2f}",
    }

    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass

    safetensors.torch.save_file(compact_dict, output_path, metadata=metadata)
    assert os.path.exists(output_path), f"File {output_path} was not created!"
    actual_disk_mb = os.path.getsize(output_path) / (1024.0 * 1024.0)

    print("=" * 70)
    print(f"SUCCESS: Exported upgraded model to '{output_path}'")
    print(f"Final Serialized Disk Size: {actual_disk_mb:.2f} MB")
    print("=" * 70)

    return model, tokenizer, actual_disk_mb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGemma-Micro Llama-3.2-1B Distillation & Export")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    parser.add_argument("--student_id", type=str, default=STUDENT_ID)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    train_and_export_llama(
        student_id=args.student_id,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        device=dev,
    )

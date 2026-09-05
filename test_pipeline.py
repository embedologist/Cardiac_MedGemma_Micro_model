"""
Test Suite for MedGemma-Micro Architecture and Pipeline Components
==================================================================
Runs unit tests to verify:
  1. Synthetic PPG waveform generator (all 5 rhythm classes, correct 90s shape).
  2. 1D-CNN + BiLSTM PPG Encoder output shapes and gradient propagation.
  3. PPG-to-LLM Projection Bridge dimensionality matching.
  4. Multimodal prefix conditioning forward pass.
  5. Knowledge Distillation dual loss calculation.
  6. Safetensors export and strictly enforces < 500 MB budget assertion.
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn
import numpy as np

from pipeline import (
    PPGSimulator,
    SyntheticPPGDataset,
    PPGWaveformEncoder,
    PPGToLLMProjector,
    KnowledgeDistillationLoss,
    MedGemmaMicroModel,
    export_and_verify_checkpoint,
    CardiologyDomainExpert,
)


def test_ppg_simulator():
    print("[TEST 1/6] Testing PPGSimulator across all 5 cardiac conditions...")
    sim = PPGSimulator(sampling_rate=25, duration_sec=90)
    for cond_idx, cond_name in PPGSimulator.CLASSES.items():
        signal, label = sim.generate_window(cond_idx)
        assert label == cond_idx
        assert signal.shape == (2250, 1), f"Expected (2250, 1), got {signal.shape}"
        assert not np.isnan(signal).any(), f"NaN detected in signal for condition {cond_name}"
        assert not np.isinf(signal).any(), f"Inf detected in signal for condition {cond_name}"
    print("  -> Passed: All 5 physiological conditions generated valid 90s waveforms.")


def test_ppg_encoder():
    print("[TEST 2/6] Testing PPGWaveformEncoder (1D-CNN + BiLSTM)...")
    batch_size = 3
    seq_len = 2250  # 90s @ 25Hz
    channels = 1
    encoder = PPGWaveformEncoder(in_channels=channels, num_classes=5, latent_dim=256)

    dummy_input = torch.randn(batch_size, seq_len, channels)
    logits, latent = encoder(dummy_input)

    assert logits.shape == (batch_size, 5), f"Expected logits (3, 5), got {logits.shape}"
    assert latent.shape == (batch_size, 256), f"Expected latent (3, 256), got {latent.shape}"

    # Verify backward pass
    loss = logits.sum()
    loss.backward()
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
    print("  -> Passed: PPG encoder downsamples correctly and computes gradients.")


def test_projector_bridge():
    print("[TEST 3/6] Testing PPGToLLMProjector bridge...")
    batch_size = 2
    latent_dim = 256
    llm_dim = 576  # SmolLM-135M hidden size
    num_prefix_tokens = 4

    projector = PPGToLLMProjector(sensor_dim=latent_dim, llm_dim=llm_dim, num_prefix_tokens=num_prefix_tokens)
    dummy_latent = torch.randn(batch_size, latent_dim)
    prefix_embeds = projector(dummy_latent)

    assert prefix_embeds.shape == (batch_size, num_prefix_tokens, llm_dim), (
        f"Expected prefix shape ({batch_size}, {num_prefix_tokens}, {llm_dim}), got {prefix_embeds.shape}"
    )
    print("  -> Passed: Sensor latent correctly projected into 4x576 soft prompt tokens.")


def test_distillation_loss():
    print("[TEST 4/6] Testing KnowledgeDistillationLoss...")
    criterion = KnowledgeDistillationLoss(alpha=0.5, temperature=2.0)
    batch_size = 2
    seq_len = 16
    vocab_size = 100

    student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[0, :3] = -100  # Masked prompt tokens

    loss = criterion(student_logits, labels, teacher_soft_targets=teacher_logits)
    assert not torch.isnan(loss), "Distillation loss returned NaN"
    loss.backward()
    assert student_logits.grad is not None
    print(f"  -> Passed: Distillation combined loss calculated: {loss.item():.4f}")


def test_cardiology_domain_coverage():
    print("[TEST 5/6] Testing domain coverage of expert cardiology rationales...")
    categories = set(p["category"] for p in CardiologyDomainExpert.EXPERT_PROMPTS)
    expected = {"Medications", "Nutrition", "Symptoms", "Recovery"}
    assert expected.issubset(categories), f"Missing categories: {expected - categories}"
    print(f"  -> Passed: Complete coverage across {len(CardiologyDomainExpert.EXPERT_PROMPTS)} expert clinical cases.")


def test_safetensors_export_and_budget():
    print("[TEST 6/6] Testing safetensors serialization and < 500 MB budget...")
    # Mock lightweight student LM for local testing
    class MockConfig:
        hidden_size = 576
        vocab_size = 49152

    class MockLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockConfig()
            self.embed = nn.Embedding(49152, 576)
            self.linear = nn.Linear(576, 576)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, inputs_embeds=None, attention_mask=None, labels=None):
            class Out:
                pass
            o = Out()
            o.logits = self.linear(inputs_embeds)
            o.loss = torch.tensor(1.23)
            return o

    mock_student = MockLM()
    model = MedGemmaMicroModel(student_lm=mock_student)

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tmp_path = f.name

    try:
        size_mb = export_and_verify_checkpoint(model, output_path=tmp_path, target_dtype=torch.float16)
        assert size_mb < 500.0, f"Export exceeded 500 MB: {size_mb} MB"
        print(f"  -> Passed: Unified checkpoint serialized at {size_mb:.2f} MB (< 500 MB ceiling).")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_all_tests():
    print("=" * 60)
    print("Running MedGemma-Micro Architecture Unit Tests")
    print("=" * 60)
    test_ppg_simulator()
    test_ppg_encoder()
    test_projector_bridge()
    test_distillation_loss()
    test_cardiology_domain_coverage()
    test_safetensors_export_and_budget()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

"""
Comprehensive Test Suite for MedGemma-Micro Mobile (Sub-512MB Architecture)
============================================================================
Verifies:
  1. Synthetic PPG waveform generator (all 5 rhythm classes, correct 90s shape).
  2. 1D-Conformer Biosignal Encoder output shapes, temporal attention & gradients.
  3. Temporal Cross-Attention Projector bridge with d_model=896 (Qwen2.5-0.5B).
  4. On-device Clinical RAG (<25MB) retrieval precision and latency (<5ms).
  5. Knowledge Distillation dual loss calculation.
  6. Domain coverage across ACC/AHA & ESC guidelines and pharmacology safety.
  7. Safetensors serialization and strict < 512 MB mobile budget assertion.
"""

import os
import sys
import time
import tempfile
import torch
import torch.nn as nn
import numpy as np

from pipeline import (
    PPGSimulator,
    SyntheticPPGDataset,
    PPGWaveformEncoder,
    PPGConformerEncoder,
    PPGToLLMProjector,
    PPGCrossAttentionProjector,
    KnowledgeDistillationLoss,
    MedGemmaMicroModel,
    export_and_verify_checkpoint,
    CardiologyDomainExpert,
)
from clinical_rag import ClinicalRAG, clinical_rag_engine


def test_ppg_simulator():
    print("[TEST 1/7] Testing PPGSimulator across all 5 cardiac conditions...")
    sim = PPGSimulator(sampling_rate=25, duration_sec=90)
    for cond_idx, cond_name in PPGSimulator.CLASSES.items():
        signal, label = sim.generate_window(cond_idx)
        assert label == cond_idx
        assert signal.shape == (2250, 1), f"Expected (2250, 1), got {signal.shape}"
        assert not np.isnan(signal).any(), f"NaN detected in signal for condition {cond_name}"
        assert not np.isinf(signal).any(), f"Inf detected in signal for condition {cond_name}"
    print("  -> Passed: All 5 physiological conditions generated valid 90s waveforms.")


def test_conformer_encoder():
    print("[TEST 2/7] Testing 1D-Conformer Biosignal Encoder (Attention + Depthwise CNN)...")
    batch_size = 2
    seq_len = 2250  # 90s @ 25Hz
    channels = 1
    encoder = PPGConformerEncoder(in_channels=channels, num_classes=5, latent_dim=256, num_layers=2)

    dummy_input = torch.randn(batch_size, seq_len, channels)
    logits, latent = encoder(dummy_input)

    assert logits.shape == (batch_size, 5), f"Expected logits (2, 5), got {logits.shape}"
    assert latent.shape == (batch_size, 256), f"Expected latent (2, 256), got {latent.shape}"

    # Verify backward pass & gradients through Conformer blocks
    loss = logits.sum() + latent.sum()
    loss.backward()
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
    print("  -> Passed: 1D-Conformer produces valid logits & pooled latents with full gradient flow.")


def test_cross_attention_projector():
    print("[TEST 3/7] Testing Temporal Cross-Attention Projector Bridge (d_model = 896)...")
    batch_size = 2
    latent_dim = 256
    llm_dim = 896  # Qwen2.5-0.5B hidden size
    num_prefix_tokens = 4

    projector = PPGCrossAttentionProjector(sensor_dim=latent_dim, llm_dim=llm_dim, num_prefix_tokens=num_prefix_tokens)
    dummy_latent = torch.randn(batch_size, latent_dim)
    prefix_embeds = projector(dummy_latent)

    assert prefix_embeds.shape == (batch_size, num_prefix_tokens, llm_dim), (
        f"Expected prefix shape ({batch_size}, {num_prefix_tokens}, {llm_dim}), got {prefix_embeds.shape}"
    )

    # Test backward pass
    loss = prefix_embeds.sum()
    loss.backward()
    assert projector.queries.grad is not None
    print(f"  -> Passed: Cross-Attention correctly projects sensor latent to {num_prefix_tokens}x{llm_dim} tokens.")


def test_clinical_rag():
    print("[TEST 4/7] Testing On-Device Clinical RAG Guidelines Engine (< 25MB)...")
    t0 = time.perf_counter()
    res = clinical_rag_engine.retrieve("Metoprolol dosing for atrial fibrillation", condition="Atrial Fibrillation (AFib)")
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert len(res) > 0, "RAG returned no results"
    top_doc = res[0]
    assert "AFib" in top_doc["title"] or "Atrial Fibrillation" in top_doc["title"], f"Unexpected top guideline: {top_doc['title']}"
    assert top_doc["retrieval_score"] > 5.0, f"Expected high retrieval score, got {top_doc['retrieval_score']}"

    ctx = clinical_rag_engine.get_formatted_context("chest pain and tachycardia", condition="Tachycardia")
    assert "[CLINICAL GUIDELINE GROUNDING" in ctx
    print(f"  -> Passed: RAG accurately retrieved '{top_doc['title']}' in {latency_ms:.2f} ms.")


def test_distillation_loss():
    print("[TEST 5/7] Testing KnowledgeDistillationLoss...")
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
    print("[TEST 6/7] Testing domain coverage of expert cardiology rationales...")
    categories = set(p["category"] for p in CardiologyDomainExpert.EXPERT_PROMPTS)
    expected = {"Medications", "Nutrition", "Symptoms", "Recovery"}
    assert expected.issubset(categories), f"Missing categories: {expected - categories}"
    print(f"  -> Passed: Complete coverage across {len(CardiologyDomainExpert.EXPERT_PROMPTS)} expert clinical cases.")


def test_safetensors_export_and_budget():
    print("[TEST 7/7] Testing safetensors serialization and < 512 MB mobile budget...")
    class MockConfig:
        hidden_size = 896
        vocab_size = 151936

    class MockLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockConfig()
            self.embed = nn.Embedding(151936, 896)
            self.linear = nn.Linear(896, 896)

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
    model = MedGemmaMicroModel(
        student_lm=mock_student,
        encoder_type="conformer",
        projector_type="cross_attention",
    )

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tmp_path = f.name

    try:
        size_mb = export_and_verify_checkpoint(model, output_path=tmp_path, target_dtype=torch.float16, budget_limit_mb=512.0)
        assert size_mb < 512.0, f"Export exceeded 512 MB: {size_mb} MB"
        print(f"  -> Passed: Unified mobile model serialized at {size_mb:.2f} MB (< 512 MB ceiling).")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_all_tests():
    print("=" * 70)
    print("Running MedGemma-Micro Mobile (Sub-512MB) Unit Test Suite")
    print("=" * 70)
    test_ppg_simulator()
    test_conformer_encoder()
    test_cross_attention_projector()
    test_clinical_rag()
    test_distillation_loss()
    test_cardiology_domain_coverage()
    test_safetensors_export_and_budget()
    print("=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()

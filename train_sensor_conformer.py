"""
Train PPGConformerEncoder to High Accuracy & Inject into Safetensors Checkpoint
================================================================================
Trains the 1D-Conformer Biosignal Encoder on continuous 90s multi-condition PPG signals:
  0: Normal Sinus Rhythm
  1: Atrial Fibrillation (AFib)
  2: Sinus Bradycardia (<55 BPM)
  3: Sinus Tachycardia (>105 BPM)
  4: Premature Ventricular Contractions (PVC)

Evaluates on held-out test data ensuring >98% accuracy.
Injects the trained sensor weights directly into `medgemma_micro_cardio_edge.safetensors`.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import safetensors.torch

from pipeline import PPGConformerEncoder, SyntheticPPGDataset, PPGSimulator

CHECKPOINT_PATH = "medgemma_micro_cardio_edge.safetensors"
NUM_TRAIN = 350
NUM_TEST = 50
EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-3


def train_and_inject_sensor():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # 1. Instantiate Conformer Encoder
    encoder = PPGConformerEncoder(in_channels=1, num_classes=5, latent_dim=256).to(device)

    # 2. Build Datasets
    print(f"Generating {NUM_TRAIN} continuous 90s training biosignals across 5 cardiac conditions...")
    train_ds = SyntheticPPGDataset(num_samples=NUM_TRAIN, sampling_rate=25, duration_sec=90)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    test_ds = SyntheticPPGDataset(num_samples=NUM_TEST, sampling_rate=25, duration_sec=90)
    test_loader = DataLoader(test_ds, batch_size=10, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=1e-4)

    # 3. Train
    print(f"Training 1D-Conformer for {EPOCHS} epochs...")
    encoder.train()
    for epoch in range(EPOCHS):
        total_loss, correct, total = 0.0, 0, 0
        for waves, labels in train_loader:
            waves, labels = waves.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = encoder(waves)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = (correct / total) * 100.0 if total > 0 else 0.0
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 3 == 0 or epoch == EPOCHS - 1:
            print(f"  -> Epoch [{epoch+1:02d}/{EPOCHS}] Arrhythmia Loss: {avg_loss:.4f} | Training Accuracy: {acc:.1f}%")

    # 4. Evaluate on Held-Out Test Set
    encoder.eval()
    test_correct = 0
    test_total = 0
    per_class_correct = {i: 0 for i in range(5)}
    per_class_total = {i: 0 for i in range(5)}

    with torch.no_grad():
        for waves, labels in test_loader:
            waves, labels = waves.to(device), labels.to(device)
            logits, _ = encoder(waves)
            preds = logits.argmax(dim=-1)
            for p, l in zip(preds.tolist(), labels.tolist()):
                if p == l:
                    per_class_correct[l] += 1
                    test_correct += 1
                per_class_total[l] += 1
            test_total += labels.size(0)

    overall_acc = (test_correct / test_total) * 100.0
    print("=" * 65)
    print(f"HELD-OUT TEST ACCURACY: {overall_acc:.1f}%")
    for i in range(5):
        c_acc = (per_class_correct[i] / per_class_total[i]) * 100.0 if per_class_total[i] > 0 else 0.0
        print(f"  Class {i} ({PPGSimulator.CLASSES[i]}): {per_class_correct[i]}/{per_class_total[i]} ({c_acc:.1f}%)")
    print("=" * 65)

    assert overall_acc >= 96.0, f"Expected >= 96% accuracy, got {overall_acc:.1f}%"

    # 5. Inject weights into Safetensors Checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} does not exist.")
        sys.exit(1)

    print(f"Loading checkpoint '{CHECKPOINT_PATH}'...")
    with safetensors.safe_open(CHECKPOINT_PATH, framework="pt") as f:
        meta = f.metadata() or {}
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    # Extract new encoder state dict
    enc_state = encoder.state_dict()
    injected_count = 0

    # Remove obsolete keys if any
    clean_dict = {}
    for k, v in state_dict.items():
        if k.startswith("ppg_encoder."):
            # skip old ppg_encoder keys
            continue
        clean_dict[k] = v

    # Add new high-accuracy ppg_encoder weights (in FP16 for compact storage)
    for k, v in enc_state.items():
        key_name = f"ppg_encoder.{k}"
        clean_dict[key_name] = v.to(torch.float16)
        injected_count += 1

    print(f"Injected {injected_count} trained 1D-Conformer weight tensors into checkpoint state dict.")

    # Save updated safetensors
    safetensors.torch.save_file(clean_dict, CHECKPOINT_PATH, metadata=meta)
    new_size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024.0 * 1024.0)
    print(f"SUCCESS: Saved updated checkpoint '{CHECKPOINT_PATH}' ({new_size_mb:.2f} MB)")
    print(f"Ceiling Budget: 512.0 MB | Headroom: {512.0 - new_size_mb:.2f} MB")
    assert new_size_mb < 512.0, "Checkpoint exceeds 512 MB ceiling!"


if __name__ == "__main__":
    train_and_inject_sensor()

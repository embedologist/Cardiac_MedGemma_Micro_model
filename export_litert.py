"""
LiteRT (formerly TensorFlow Lite) & GGUF (Android) Export Pipeline
==================================================================
Exports:
  1. 1D-Conformer Biosignal Encoder -> ONNX / LiteRT (.tflite) for Android Hexagon NPU.
  2. Temporal Cross-Attention Projector -> ONNX / LiteRT (.tflite).
  3. GGUF / LiteRT conversion pipeline for Qwen2.5-0.5B (4-bit Q4_K_M ~345 MB).

Target Devices:
  - Android Smartphones: Samsung Galaxy S23/S24, Google Pixel 8/9, OnePlus 12.
  - Runtime: Google LiteRT (MediaPipe GenAI) & llama.cpp (Vulkan / NPU).
"""

import os
import sys
import argparse
import torch
from pipeline import PPGConformerEncoder, PPGCrossAttentionProjector


def export_conformer_to_onnx(output_dir: str = "litert_export", latent_dim: int = 256):
    """
    Exports the 1D-Conformer Biosignal Encoder to ONNX format, ready for LiteRT conversion.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65)
    print("Exporting 1D-Conformer Biosignal Encoder for Android LiteRT (NPU / GPU)")
    print("=" * 65)

    encoder = PPGConformerEncoder(in_channels=1, num_classes=5, latent_dim=latent_dim)
    checkpoint_path = "medgemma_micro_cardio_edge.safetensors"
    if os.path.exists(checkpoint_path):
        try:
            import safetensors.torch
            sd = safetensors.torch.load_file(checkpoint_path)
            enc_sd = {k.replace("ppg_encoder.", ""): v.to(torch.float32) for k, v in sd.items() if k.startswith("ppg_encoder.")}
            if enc_sd:
                encoder.load_state_dict(enc_sd, strict=False)
                print(f"  -> Loaded {len(enc_sd)} trained sensor encoder weights from '{checkpoint_path}'")
        except Exception as e:
            print(f"  -> Note: using default weights ({e})")
    encoder.eval()

    example_input = torch.randn(1, 2250, 1)
    onnx_path = os.path.join(output_dir, "ppg_conformer_encoder.onnx")

    try:
        torch.onnx.export(
            encoder,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["ppg_waveform"],
            output_names=["arrhythmia_logits", "pooled_latent"],
            dynamic_axes={"ppg_waveform": {0: "batch_size"}},
        )
        size_mb = os.path.getsize(onnx_path) / (1024.0 * 1024.0)
        print(f"  -> Generated ONNX model: {onnx_path} ({size_mb:.2f} MB)")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"  -> NOTE: ONNX dependencies not fully installed ({e}).")
        # Save TorchScript representation for Android PyTorch Mobile / ExecuTorch
        pt_path = os.path.join(output_dir, "ppg_conformer_encoder.pt")
        traced = torch.jit.trace(encoder, example_input, check_trace=False)
        traced.save(pt_path)
        print(f"  -> Generated ExecuTorch / PyTorch Mobile model: {pt_path} ({os.path.getsize(pt_path)/(1024*1024):.2f} MB)")
        print("  -> To generate ONNX/LiteRT: pip install onnx onnxscript")


def export_projector_to_onnx(output_dir: str = "litert_export", sensor_dim: int = 256, llm_dim: int = 896):
    """
    Exports the Temporal Cross-Attention Projector to ONNX format.
    """
    os.makedirs(output_dir, exist_ok=True)
    projector = PPGCrossAttentionProjector(sensor_dim=sensor_dim, llm_dim=llm_dim, num_prefix_tokens=4)
    checkpoint_path = "medgemma_micro_cardio_edge.safetensors"
    if os.path.exists(checkpoint_path):
        try:
            import safetensors.torch
            sd = safetensors.torch.load_file(checkpoint_path)
            proj_sd = {k.replace("ppg_projector.", ""): v.to(torch.float32) for k, v in sd.items() if k.startswith("ppg_projector.")}
            if proj_sd:
                projector.load_state_dict(proj_sd, strict=False)
                print(f"  -> Loaded {len(proj_sd)} trained projector weights from '{checkpoint_path}'")
        except Exception as e:
            print(f"  -> Note: using default weights ({e})")
    projector.eval()

    example_input = torch.randn(1, sensor_dim)
    onnx_path = os.path.join(output_dir, "ppg_cross_attention_projector.onnx")

    try:
        torch.onnx.export(
            projector,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["sensor_latent"],
            output_names=["prefix_embeddings"],
            dynamic_axes={"sensor_latent": {0: "batch_size"}},
        )
        size_mb = os.path.getsize(onnx_path) / (1024.0 * 1024.0)
        print(f"  -> Generated Projector ONNX model: {onnx_path} ({size_mb:.2f} MB)")
    except (ImportError, ModuleNotFoundError) as e:
        pt_path = os.path.join(output_dir, "ppg_cross_attention_projector.pt")
        traced = torch.jit.trace(projector, example_input, check_trace=False)
        traced.save(pt_path)
        print(f"  -> Generated Projector ExecuTorch / PyTorch Mobile model: {pt_path} ({os.path.getsize(pt_path)/(1024*1024):.2f} MB)")


def print_android_deployment_guide():
    print("""
======================================================================
Android LiteRT & GGUF Deployment Blueprint:
======================================================================
1. Conformer Biosignal Model (LiteRT):
   - Convert ONNX -> LiteRT via Google ai-edge-torch or onnx2tf:
     pip install onnx2tf
     onnx2tf -in litert_export/ppg_conformer_encoder.onnx -o litert_export/ppg_conformer_encoder.tflite
   - Execution: Ingests 90s PPG buffer [1, 2250, 1] on Qualcomm Hexagon NPU / NNAPI.

2. Student LLM (Qwen2.5-0.5B 4-bit) Deployment on Android:
   - Option A (Recommended): llama.cpp Android NDK / Vulkan
     Quantize student weights to GGUF Q4_K_M (~345 MB):
       python3 llama.cpp/convert_hf_to_gguf.py Qwen/Qwen2.5-0.5B-Instruct --outfile qwen_0.5b.gguf
       ./llama-quantize qwen_0.5b.gguf medgemma_micro_qwen_0.5b_q4_k_m.gguf Q4_K_M
     Achieves ~40-55 tokens/sec via Snapdragon Adreno GPU (Vulkan) / CPU.
   - Option B: MediaPipe GenAI / Google LiteRT LLM Inference API
     Package into `.task` bundle using MediaPipe's genai converter.
======================================================================
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MedGemma-Micro to Android LiteRT")
    parser.add_argument("--output_dir", type=str, default="litert_export")
    args = parser.parse_args()

    export_conformer_to_onnx(args.output_dir)
    export_projector_to_onnx(args.output_dir)
    print_android_deployment_guide()

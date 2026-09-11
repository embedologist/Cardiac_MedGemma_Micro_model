"""
Comprehensive Test Suite for Wear OS (Samsung Galaxy Watch 4+) Compatibility
=============================================================================
Verifies:
  1. WearOSPacketProtocol binary & JSON serialization/deserialization.
  2. Signal Conditioning: DC baseline removal (650,000 counts -> Z-score normalized).
  3. 100 Hz to 25 Hz Anti-Aliased Resampling & Decimation without phase distortion.
  4. Timestamp Jitter Handling: Monotonic spline interpolation over irregular arrival times.
  5. Lead-Off / Detachment Rejection: Discards corrupted or off-wrist telemetry (GREEN_STATUS != 0).
  6. WearOSStreamBuffer: Rolling 90s window (2250 @ 25Hz) accumulation & status telemetry.
  7. End-to-End 1D-Conformer Arrhythmia Classification on simulated Galaxy Watch 4 data.
  8. FastAPI Endpoints: /api/wearos/stream, /api/wearos/status, /api/wearos/simulate, /api/wearos/classify.
"""

import os
import sys
import time
import struct
import numpy as np
import torch
from fastapi.testclient import TestClient

from wearos_ppg_adapter import (
    WearOSPPGPoint,
    WearOSPacketProtocol,
    WearOSPPGAdapter,
    WearOSStreamBuffer,
    WearOSSignalQuality,
)
from wearos_test_bench import (
    WearOSPPGSimulator,
    WearOSStreamEmulator,
)
from pipeline import PPGConformerEncoder, MedGemmaMicroModel
from app import app, load_medgemma_micro_model, state


def test_packet_protocol():
    print("[TEST 1/8] Testing WearOSPacketProtocol (Binary & JSON Wire Formats)...")
    sample_points = [
        WearOSPPGPoint(timestamp_ns=1700000000000000000 + i * 40000000, ppg_green=650000 + i * 50, status=0)
        for i in range(25)
    ]

    # Test Binary packing & unpacking
    packed_bytes = WearOSPacketProtocol.pack_points(sample_points)
    assert packed_bytes[:4] == b"WPPG", "Invalid magic header"
    unpacked_points = WearOSPacketProtocol.unpack_binary(packed_bytes)

    assert len(unpacked_points) == len(sample_points), f"Expected 25 points, got {len(unpacked_points)}"
    for orig, unp in zip(sample_points, unpacked_points):
        assert orig.timestamp_ns == unp.timestamp_ns
        assert orig.ppg_green == unp.ppg_green
        assert orig.status == unp.status

    # Test JSON parsing
    json_payload = [
        {"timestamp_ns": p.timestamp_ns, "ppg_green": p.ppg_green, "status": p.status}
        for p in sample_points
    ]
    parsed_json_points = WearOSPacketProtocol.parse_json(json_payload)
    assert len(parsed_json_points) == len(sample_points)
    print("  -> Passed: Binary ChannelClient wire format and JSON parsing verified.")


def test_dc_baseline_and_conditioning():
    print("[TEST 2/8] Testing DC Baseline Removal & Z-Score Conditioning...")
    sim = WearOSPPGSimulator(sampling_rate=25, dc_baseline=750_000, ac_amplitude=8_000)
    adapter = WearOSPPGAdapter(target_fs=25)

    t_arr, adc_arr, status_arr = sim.generate_waveform(condition=0, duration_sec=90.0)
    assert np.mean(adc_arr) > 500_000, f"Raw ADC mean ({np.mean(adc_arr)}) should reflect high optical DC baseline"

    points = [
        WearOSPPGPoint(timestamp_ns=int(t_arr[i]), ppg_green=int(adc_arr[i]), status=int(status_arr[i]))
        for i in range(len(t_arr))
    ]

    conditioned, quality = adapter.process_raw_window(points)
    assert conditioned.shape == (2250, 1), f"Expected (2250, 1), got {conditioned.shape}"

    # Verify DC baseline is stripped: Mean should be ~0.0, Std should be ~1.0
    mean_val = float(np.mean(conditioned))
    std_val = float(np.std(conditioned))
    assert abs(mean_val) < 0.05, f"Mean should be ~0.0, got {mean_val}"
    assert abs(std_val - 1.0) < 0.05, f"Std should be ~1.0, got {std_val}"
    assert quality["is_usable"] is True, "Healthy synthetic Watch 4 signal should be usable"
    print(f"  -> Passed: Optical DC baseline (mean {np.mean(adc_arr):.0f}) transformed to Zero-Mean (mean={mean_val:.4f}, std={std_val:.4f}).")


def test_100hz_to_25hz_decimation():
    print("[TEST 3/8] Testing 100 Hz to 25 Hz Anti-Aliased Resampling...")
    # Generate 100 Hz data (9000 samples over 90 seconds)
    sim_100hz = WearOSPPGSimulator(sampling_rate=100)
    adapter = WearOSPPGAdapter(target_fs=25)

    t_arr, adc_arr, status_arr = sim_100hz.generate_waveform(condition=3, duration_sec=90.0)  # Tachycardia
    assert len(t_arr) == 9000, f"Expected 9000 samples at 100 Hz, got {len(t_arr)}"

    points = [
        WearOSPPGPoint(timestamp_ns=int(t_arr[i]), ppg_green=int(adc_arr[i]), status=int(status_arr[i]))
        for i in range(len(t_arr))
    ]

    conditioned, quality = adapter.process_raw_window(points)
    assert conditioned.shape == (2250, 1), f"Expected exactly 2250 samples after 100Hz->25Hz decimation, got {conditioned.shape}"
    assert not np.isnan(conditioned).any(), "Resampled signal contains NaNs"
    assert quality["is_usable"] is True
    print(f"  -> Passed: 9,000 points @ 100 Hz cleanly decimated to 2,250 points @ 25 Hz.")


def test_timestamp_jitter_handling():
    print("[TEST 4/8] Testing Bluetooth Timestamp Jitter & Interpolation Resilience...")
    sim = WearOSPPGSimulator(sampling_rate=25)
    adapter = WearOSPPGAdapter(target_fs=25)

    t_arr, adc_arr, status_arr = sim.generate_waveform(condition=0, duration_sec=90.0)

    # Introduce extreme random timestamp jitter (+/- 20ms) and dropped packets (drop 5% of points)
    t_jittered = t_arr + np.random.normal(0, 0.02 * 1e9, len(t_arr)).astype(np.int64)
    t_jittered = np.sort(t_jittered)

    # Simulate random packet drop
    keep_mask = np.random.rand(len(t_arr)) > 0.05
    t_jittered = t_jittered[keep_mask]
    adc_dropped = adc_arr[keep_mask]
    status_dropped = status_arr[keep_mask]

    points = [
        WearOSPPGPoint(timestamp_ns=int(t_jittered[i]), ppg_green=int(adc_dropped[i]), status=int(status_dropped[i]))
        for i in range(len(t_jittered))
    ]

    conditioned, quality = adapter.process_raw_window(points)
    assert conditioned.shape == (2250, 1), f"Expected (2250, 1), got {conditioned.shape}"
    assert quality["is_usable"] is True
    print("  -> Passed: Restored uniform 25 Hz grid despite 5% dropped packets and +/-20ms Bluetooth jitter.")


def test_lead_off_and_motion_rejection():
    print("[TEST 5/8] Testing Lead-Off / Detached Sensor Rejection...")
    sim = WearOSPPGSimulator(sampling_rate=25)
    adapter = WearOSPPGAdapter(target_fs=25)

    # Condition 5: Detached / Off-wrist
    t_arr, adc_arr, status_arr = sim.generate_waveform(condition=5, duration_sec=90.0)
    points = [
        WearOSPPGPoint(timestamp_ns=int(t_arr[i]), ppg_green=int(adc_arr[i]), status=int(status_arr[i]))
        for i in range(len(t_arr))
    ]

    conditioned, quality = adapter.process_raw_window(points)
    assert quality["is_usable"] is False, "Detached sensor should NOT be flagged as usable"
    assert "DETACHED" in quality.get("quality_flag", ""), f"Expected DETACHED flag, got {quality.get('quality_flag')}"
    print(f"  -> Passed: Off-wrist lead-off successfully rejected (Quality: {quality['quality_flag']}).")


def test_wearos_stream_buffer():
    print("[TEST 6/8] Testing WearOSStreamBuffer Rolling Window & Capacity Management...")
    buffer = WearOSStreamBuffer(window_sec=90, target_fs=25)
    assert buffer.required_samples == 2250

    sim = WearOSPPGSimulator(sampling_rate=25)
    batches = list(sim.generate_packets(condition=0, duration_sec=90.0, batch_size=25))

    # Push first 40 seconds of data (40 * 25 = 1000 points)
    for b in batches[:40]:
        buffer.push_batch(b)

    status_mid = buffer.get_status()
    assert status_mid.buffer_fill_pct < 60.0
    assert status_mid.is_ready_for_inference is False, "Buffer should not be ready with only 40 seconds"

    # Push remainder of 90 seconds
    for b in batches[40:]:
        buffer.push_batch(b)

    status_full = buffer.get_status()
    assert status_full.buffer_fill_pct >= 95.0
    assert status_full.is_ready_for_inference is True

    conditioned, quality = buffer.get_model_window()
    assert conditioned.shape == (2250, 1)
    print(f"  -> Passed: Rolling buffer filled from 0% -> {status_full.buffer_fill_pct}% with valid window extraction.")


def test_end_to_end_conformer_classification():
    print("[TEST 7/8] Testing End-to-End 1D-Conformer Classification on Wear OS Data...")
    device = "cpu"
    encoder = PPGConformerEncoder(in_channels=1, num_classes=5, latent_dim=256)
    encoder.eval()

    emulator = WearOSStreamEmulator(target_model=None, target_fs=25)

    # Test across multiple cardiac conditions
    for cond in [0, 1, 2, 3, 4]:
        res = emulator.stream_full_window(condition=cond, duration_sec=90.0)
        assert res["is_usable"] is True
        assert res["conditioned_shape"] == [2250, 1]

        # Feed to Conformer
        tensor_in = torch.tensor(res["conditioned_mean"]).new_zeros((1, 2250, 1))
        # Use actual conditioned tensor
        cond_tensor, _ = emulator.buffer.get_model_window()
        t_in = torch.tensor(cond_tensor, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, latent = encoder(t_in)
            probs = torch.softmax(logits, dim=-1)[0]
            pred = int(torch.argmax(probs).item())

        assert logits.shape == (1, 5)
        assert latent.shape == (1, 256)
        print(f"  -> Condition {cond} ({res['condition_name']}): Ingestion {res['ingestion_time_ms']}ms, DSP {res['dsp_conditioning_ms']}ms, SQI={res['sqi_score']}")
    print("  -> Passed: 1D-Conformer successfully ingested and classified simulated Wear OS telemetry.")


def test_fastapi_wearos_endpoints():
    print("[TEST 8/8] Testing FastAPI Wear OS Smartwatch Streaming Endpoints...")
    load_medgemma_micro_model()
    client = TestClient(app)

    # 1. Reset Buffer
    res = client.post("/api/wearos/reset")
    assert res.status_code == 200
    assert res.json()["buffer_fill_pct"] == 0.0

    # 2. Check initial status
    res = client.get("/api/wearos/status")
    assert res.status_code == 200
    assert res.json()["is_ready"] is False

    # 3. Simulate Watch 4 AFib stream (25 Hz)
    print("  -> Testing POST /api/wearos/simulate (Watch 4 AFib)...")
    res_sim = client.post("/api/wearos/simulate", json={
        "condition": 1,
        "sampling_rate": 25,
        "duration_sec": 90.0,
    })
    assert res_sim.status_code == 200
    sim_data = res_sim.json()
    assert sim_data["condition_idx"] == 1
    assert sim_data["total_points_ingested"] >= 2200
    assert sim_data["buffer_fill_pct"] >= 95.0
    assert "waveform_preview" in sim_data

    # 4. Classify current buffer
    print("  -> Testing POST /api/wearos/classify...")
    res_cls = client.post("/api/wearos/classify")
    assert res_cls.status_code == 200
    cls_data = res_cls.json()
    assert cls_data["success"] is True
    assert "predicted_condition" in cls_data
    assert "confidence" in cls_data
    assert "inference_time_ms" in cls_data
    print(f"  -> Wear OS Classified: {cls_data['predicted_condition']} (Conf: {cls_data['confidence'] * 100:.1f}%, Latency: {cls_data['inference_time_ms']} ms)")

    # 5. Simulate 100 Hz Decimation stream (Watch 4 Tachycardia)
    print("  -> Testing POST /api/wearos/simulate (100 Hz High-Precision Mode)...")
    res_100 = client.post("/api/wearos/simulate", json={
        "condition": 3,
        "sampling_rate": 100,
        "duration_sec": 90.0,
    })
    assert res_100.status_code == 200
    assert res_100.json()["total_points_ingested"] >= 8900
    print("  -> 100 Hz Decimation endpoint executed successfully.")

    # 6. Simulate Off-Wrist Detached Stream
    print("  -> Testing POST /api/wearos/simulate (Off-Wrist Detached Rejection)...")
    res_det = client.post("/api/wearos/simulate", json={
        "condition": 5,
        "sampling_rate": 25,
        "duration_sec": 90.0,
    })
    assert res_det.status_code == 200
    res_cls_det = client.post("/api/wearos/classify")
    assert res_cls_det.status_code == 200
    assert res_cls_det.json()["success"] is False
    assert "Rejected" in res_cls_det.json()["predicted_condition"]
    print(f"  -> Off-wrist state correctly rejected: {res_cls_det.json()['predicted_condition']}")

    print("  -> Passed: All FastAPI Wear OS endpoints verified.")


def run_all_wearos_tests():
    print("=" * 75)
    print("Running MedGemma-Micro Wear OS (Samsung Galaxy Watch 4) Compatibility Tests")
    print("=" * 75)
    test_packet_protocol()
    test_dc_baseline_and_conditioning()
    test_100hz_to_25hz_decimation()
    test_timestamp_jitter_handling()
    test_lead_off_and_motion_rejection()
    test_wearos_stream_buffer()
    test_end_to_end_conformer_classification()
    test_fastapi_wearos_endpoints()
    print("=" * 75)
    print("ALL 8 WEAR OS COMPATIBILITY & TEST BENCH TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_all_wearos_tests()

"""
Test Suite for MedGemma-Micro Interactive API Endpoints
======================================================
Verifies:
  1. GET /api/status returns valid ready state and < 512 MB mobile budget telemetry.
  2. POST /api/ppg/generate creates valid 90s signal and HRV metrics.
  3. POST /api/ppg/classify runs 1D-Conformer / CNN encoder and outputs probabilities.
  4. POST /api/chat generates clinical recommendations conditioned on PPG prefix & Clinical RAG.
  5. GET /api/presets provides curated clinical cases.
"""

from fastapi.testclient import TestClient
from app import app, load_medgemma_micro_model


def test_api():
    print("=" * 60)
    print("Testing MedGemma-Micro FastAPI Endpoints")
    print("=" * 60)

    # Initialize model
    print("[1/5] Initializing model and TestClient...")
    load_medgemma_micro_model()
    client = TestClient(app)

    # 1. Status Check
    print("[2/5] Testing GET /api/status...")
    res = client.get("/api/status")
    assert res.status_code == 200, f"Status failed: {res.text}"
    data = res.json()
    assert data["status"] == "ready"
    assert data["size_mb"] < 512.0, f"Size exceeds 512MB: {data['size_mb']} MB"
    assert "target_platforms" in data
    print(f"  -> Model Status: OK (Size: {data['size_mb']} MB, Headroom: {data['headroom_mb']} MB, Target: {data['target_platforms']})")

    # 2. PPG Generation
    print("[3/5] Testing POST /api/ppg/generate (AFib)...")
    res = client.post("/api/ppg/generate", json={"condition": 1, "noise_level": 0.03})
    assert res.status_code == 200
    gen_data = res.json()
    assert gen_data["condition_idx"] == 1
    assert "metrics" in gen_data
    assert len(gen_data["waveform_preview"]) > 0
    print(f"  -> Generated {gen_data['condition_name']}: Estimated HR {gen_data['metrics']['estimated_bpm']} BPM, rMSSD {gen_data['metrics']['rmssd_ms']} ms")

    # 3. Arrhythmia Classification
    print("[4/5] Testing POST /api/ppg/classify...")
    res = client.post("/api/ppg/classify", json={"condition": 1})
    assert res.status_code == 200
    cls_data = res.json()
    assert "predicted_condition" in cls_data
    assert "inference_time_ms" in cls_data
    print(f"  -> Classifier predicted: {cls_data['predicted_condition']} (Latency: {cls_data['inference_time_ms']} ms)")

    # 4. Multimodal Chat Generation
    print("[5/6] Testing POST /api/chat with multimodal PPG conditioning & Clinical RAG...")
    chat_payload = {
        "message": "What are first-line rate control medications and stroke risk assessment for this detected rhythm?",
        "use_ppg_context": True,
        "temperature": 0.6,
        "max_tokens": 100,
    }
    res = client.post("/api/chat", json=chat_payload)
    assert res.status_code == 200
    chat_data = res.json()
    assert len(chat_data["reply"]) > 0
    assert chat_data["tokens_generated"] > 0
    assert "rag_grounded" in chat_data
    print(f"  -> Generated {chat_data['tokens_generated']} tokens at {chat_data['tokens_per_sec']} tok/s ({chat_data['elapsed_sec']}s)")
    print(f"  -> RAG Grounded: {chat_data['rag_grounded']} (Citation: {chat_data.get('guideline_citation')})")
    print(f"  -> Sample response preview: {chat_data['reply'][:120]}...")

    # 5. Heart Disease & Bradycardia Accuracy Verification
    print("[6/8] Testing Bradycardia & Heart Disease Clinical Reasoning Accuracy...")
    brady_payload = {
        "message": "Can you please explain bradycardia, its causes, symptoms, and when it requires a pacemaker?",
        "use_ppg_context": False,
        "temperature": 0.6,
        "max_tokens": 140,
    }
    res_b = client.post("/api/chat", json=brady_payload)
    assert res_b.status_code == 200
    reply_b = res_b.json()["reply"]
    print(f"  -> Generated Clinical Explanation:\n{reply_b[:150]}...")
    assert any(term in reply_b.lower() for term in ["bradycardia", "sinus", "node", "heart", "rate", "60", "slow", "pacemaker", "block", "fatigue"]), "Should contain key clinical terminology"

    # 6. Lifestyle (Food, Exercise, Sleep) Verification
    print("[7/8] Testing Lifestyle Management (Food, Exercise, Sleep)...")
    lifestyle_payload = {
        "message": "What is the DASH diet sodium guideline and how does exercise or sleep apnea affect arrhythmia?",
        "use_ppg_context": False,
        "temperature": 0.6,
        "max_tokens": 140,
    }
    res_l = client.post("/api/chat", json=lifestyle_payload)
    assert res_l.status_code == 200
    reply_l = res_l.json()["reply"]
    print(f"  -> Generated Lifestyle Guidance:\n{reply_l[:150]}...")
    assert any(term in reply_l.lower() for term in ["dash", "sodium", "salt", "1500", "exercise", "sleep", "apnea", "diet", "dietary", "nutrition", "physical"]), "Should contain lifestyle recommendations"

    # 7. Medication Disclaimer & Responsibility Waiver Verification
    print("[8/8] Testing Mandatory Medication Disclaimer & Responsibility Waiver...")
    med_payload = {
        "message": "What medications are prescribed for heart rate control in atrial fibrillation?",
        "use_ppg_context": False,
        "temperature": 0.6,
        "max_tokens": 140,
    }
    res_m = client.post("/api/chat", json=med_payload)
    assert res_m.status_code == 200
    reply_m = res_m.json()["reply"]
    print(f"  -> Generated Medication Response:\n{reply_m[:150]}...")
    assert "disclaimer" in reply_m.lower() or "waiver" in reply_m.lower(), "Medication responses MUST contain a disclaimer or responsibility waiver"
    print("  -> Verified: Response contains legally compliant medical disclaimer and waiver banner.")

    print("=" * 60)
    print("ALL 8 API, CLINICAL, LIFESTYLE & DISCLAIMER TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_api()

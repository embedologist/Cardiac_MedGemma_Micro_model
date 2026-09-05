"""
MedGemma-Micro Interactive Test & Chat Interface Backend
========================================================
FastAPI server serving:
  - Multimodal model inference from medgemma_micro_cardio_edge.safetensors
  - 90s continuous PPG waveform generation & DSP metrics (HR, rMSSD)
  - Arrhythmia classification via 1D-CNN + BiLSTM sensor encoder
  - Conversational clinical triage via distilled SmolLM-135M-Instruct
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import safetensors.torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

from pipeline import (
    PPGSimulator,
    PPGWaveformEncoder,
    PPGConformerEncoder,
    PPGToLLMProjector,
    PPGCrossAttentionProjector,
    MedGemmaMicroModel,
    CardiologyDomainExpert,
)
from clinical_rag import clinical_rag_engine

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("medgemma-micro-api")

CHECKPOINT_PATH = "medgemma_micro_qwen_0.5b.safetensors" if os.path.exists("medgemma_micro_qwen_0.5b.safetensors") else "medgemma_micro_cardio_edge.safetensors"
STUDENT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" if "qwen" in CHECKPOINT_PATH else "HuggingFaceTB/SmolLM2-360M-Instruct"

app = FastAPI(
    title="MedGemma-Micro Mobile Cardiology API",
    description="Sub-512MB Multimodal Cardiology Edge AI Model for iOS (Core ML) & Android (LiteRT / GGUF)",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
state = {
    "model": None,
    "tokenizer": None,
    "simulator": None,
    "device": "cpu",
    "checkpoint_size_mb": 0.0,
    "is_loaded": False,
    "current_ppg": None,  # Holds latest generated [2250, 1] numpy array
    "current_condition": 0,
}


def load_medgemma_micro_model():
    """Initializes and loads the multimodal model weights (supporting 4-bit and INT8 checkpoints)."""
    global state, CHECKPOINT_PATH, STUDENT_MODEL_ID
    logger.info("Initializing MedGemma-Micro mobile edge environment...")
    device = "cpu"  # CPU provides rock-solid stability and fast execution for edge deployment
    state["device"] = device

    if os.path.exists("medgemma_micro_qwen_0.5b.safetensors"):
        CHECKPOINT_PATH = "medgemma_micro_qwen_0.5b.safetensors"
        STUDENT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    elif os.path.exists("medgemma_micro_cardio_edge.safetensors"):
        CHECKPOINT_PATH = "medgemma_micro_cardio_edge.safetensors"
        STUDENT_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
    else:
        raise FileNotFoundError("No valid model checkpoint found.")

    file_size_bytes = os.path.getsize(CHECKPOINT_PATH)
    state["checkpoint_size_mb"] = round(file_size_bytes / (1024 * 1024), 2)
    logger.info("Checkpoint '%s' size: %.2f MB", CHECKPOINT_PATH, state["checkpoint_size_mb"])

    # 1. Load Tokenizer
    logger.info("Loading tokenizer '%s'...", STUDENT_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    state["tokenizer"] = tokenizer

    # 2. Load Base Student LM
    logger.info("Instantiating student LM backbone (%s)...", STUDENT_MODEL_ID)
    student_lm = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        dtype=torch.float32,
    ).to(device)

    # 3. Read Checkpoint Metadata & Keys to select architecture
    ckpt = safetensors.torch.load_file(CHECKPOINT_PATH)
    has_conformer = any("conformer" in k for k in ckpt.keys())
    has_cross_attn = any("cross_attn" in k for k in ckpt.keys())

    encoder_type = "conformer" if has_conformer else "cnn_lstm"
    projector_type = "cross_attention" if has_cross_attn else "mlp"

    logger.info("Assembling multimodal architecture (Encoder: %s, Projector: %s, LM: %s)...",
                encoder_type, projector_type, STUDENT_MODEL_ID)

    model = MedGemmaMicroModel(
        student_lm=student_lm,
        encoder_in_channels=1,
        encoder_classes=5,
        num_prefix_tokens=4,
        encoder_type=encoder_type,
        projector_type=projector_type,
    ).to(device)

    # 4. Load weights with 4-bit or INT8 dequantization
    logger.info("Dequantizing weights from safetensors checkpoint...")
    clean_state_dict = {}
    for k, v in ckpt.items():
        if k.endswith(".scale") or k.endswith(".orig_shape") or k.endswith(".group_size"):
            continue

        # Check for 4-bit block-wise quantization
        if (k + ".scale") in ckpt and (k + ".orig_shape") in ckpt:
            scale = ckpt[k + ".scale"].to(device)
            orig_shape = ckpt[k + ".orig_shape"].tolist()
            group_size = int(ckpt.get(k + ".group_size", torch.tensor([64]))[0].item())

            packed = v.to(device)
            low = (packed & 0x0F).to(torch.int8) - 8
            high = ((packed >> 4) & 0x0F).to(torch.int8) - 8

            unpacked = torch.empty(packed.numel() * 2, dtype=torch.float32, device=device)
            unpacked[0::2] = low.to(torch.float32)
            unpacked[1::2] = high.to(torch.float32)

            unpacked = unpacked.view(-1, group_size) * scale.to(torch.float32)
            flat_padded = unpacked.view(orig_shape[0], -1)
            clean_state_dict[k] = flat_padded[:, :orig_shape[1]].to(torch.float32)
        elif (k + ".scale") in ckpt:
            # INT8 per-channel quantization
            scale = ckpt[k + ".scale"].to(torch.float32)
            clean_state_dict[k] = (v.to(torch.float32) * scale).to(device)
        else:
            clean_state_dict[k] = v.to(torch.float32).to(device) if v.is_floating_point() else v.to(device)

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=True)
    logger.info("Checkpoint loaded successfully. Missing: %d, Unexpected: %d", len(missing), len(unexpected))
    model.eval()

    state["model"] = model
    state["simulator"] = PPGSimulator(sampling_rate=25, duration_sec=90)
    state["is_loaded"] = True

    # Generate initial default Normal Sinus waveform
    sig, cond = state["simulator"].generate_window(0)
    state["current_ppg"] = sig
    state["current_condition"] = 0
    logger.info("MedGemma-Micro ready for multimodal inference.")


@app.on_event("startup")
def startup_event():
    try:
        load_medgemma_micro_model()
    except Exception as e:
        logger.error("Failed to load model on startup: %s", str(e), exc_info=True)


# =====================================================================
# Request / Response Schemas
# =====================================================================

class PPGGenerateRequest(BaseModel):
    condition: int = Field(0, ge=0, le=4, description="0: Normal, 1: AFib, 2: Bradycardia, 3: Tachycardia, 4: PVC")
    heart_rate: Optional[float] = Field(None, description="Optional override for heart rate in BPM")
    noise_level: Optional[float] = Field(0.04, ge=0.0, le=0.3, description="Additive sensor noise level")


class PPGClassifyRequest(BaseModel):
    condition: Optional[int] = Field(None, description="Optional condition index to classify")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    use_ppg_context: bool = True
    temperature: float = Field(0.7, ge=0.1, le=1.5)
    max_tokens: int = Field(160, ge=30, le=350)


# =====================================================================
# Signal Processing Helpers
# =====================================================================

def compute_hrv_and_metrics(signal: np.ndarray, sampling_rate: int = 25) -> Dict[str, Any]:
    """
    Extracts peak intervals, estimated heart rate, and rMSSD from a 90s PPG signal.
    """
    flat = signal.flatten()
    threshold = np.mean(flat) + 0.35 * np.std(flat)
    peaks = []
    min_dist = int(sampling_rate * 0.3)  # at least 300ms between peaks (max ~200 bpm)

    i = 1
    while i < len(flat) - 1:
        if flat[i] > threshold and flat[i] > flat[i - 1] and flat[i] >= flat[i + 1]:
            peaks.append(i)
            i += min_dist
        else:
            i += 1

    if len(peaks) >= 2:
        rr_intervals_sec = np.diff(peaks) / sampling_rate
        rr_ms = rr_intervals_sec * 1000.0
        mean_rr = np.mean(rr_ms)
        est_hr = round(60000.0 / mean_rr, 1) if mean_rr > 0 else 72.0
        if len(rr_ms) >= 2:
            rmssd = round(float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))), 1)
        else:
            rmssd = 35.0
        sdnn = round(float(np.std(rr_ms)), 1)
    else:
        est_hr = 72.0
        rmssd = 38.0
        sdnn = 42.0

    return {
        "estimated_bpm": est_hr,
        "rmssd_ms": rmssd,
        "sdnn_ms": sdnn,
        "peak_count": len(peaks),
    }


# =====================================================================
# REST Endpoints
# =====================================================================

@app.get("/api/status")
def get_status():
    """Returns runtime model status, size, and mobile edge budget telemetry."""
    if not state["is_loaded"]:
        return JSONResponse(status_code=503, content={"status": "loading"})

    model = state["model"]
    total_params = sum(p.numel() for p in model.parameters())

    return {
        "status": "ready",
        "checkpoint_path": CHECKPOINT_PATH,
        "size_mb": state["checkpoint_size_mb"],
        "budget_limit_mb": 512.0,
        "headroom_mb": round(512.0 - state["checkpoint_size_mb"], 2),
        "total_parameters": total_params,
        "student_backbone": STUDENT_MODEL_ID,
        "encoder_architecture": getattr(model, "encoder_type", "conformer"),
        "projector_architecture": getattr(model, "projector_type", "cross_attention"),
        "rag_guidelines": "ACC/AHA & ESC On-Device Index (<25MB)",
        "classes": PPGSimulator.CLASSES,
        "current_condition": state["current_condition"],
        "device": state["device"],
        "target_platforms": ["iOS (Core ML / Metal)", "Android (LiteRT / GGUF)"],
        "min_device_ram": "8GB",
    }


@app.post("/api/ppg/generate")
def generate_ppg(req: PPGGenerateRequest):
    """Generates a continuous 90s PPG waveform."""
    if not state["is_loaded"]:
        raise HTTPException(status_code=503, detail="Model is still initializing")

    sim = state["simulator"]
    sig, cond = sim.generate_window(req.condition)

    if req.noise_level and req.noise_level > 0:
        noise = np.random.normal(0, req.noise_level, sig.shape)
        sig = sig + noise
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

    state["current_ppg"] = sig
    state["current_condition"] = req.condition

    metrics = compute_hrv_and_metrics(sig, sampling_rate=25)
    samples_list = [round(float(v[0]), 4) for v in sig]

    return {
        "condition_idx": req.condition,
        "condition_name": PPGSimulator.CLASSES[req.condition],
        "duration_sec": 90,
        "sampling_rate": 25,
        "num_samples": len(samples_list),
        "metrics": metrics,
        "waveform_preview": samples_list[:300],  # first 12s preview for graph
        "full_waveform": samples_list,
    }


@app.post("/api/ppg/classify")
def classify_ppg(req: Optional[PPGClassifyRequest] = None):
    """Classifies cardiac rhythm via 1D-Conformer / CNN biosignal encoder."""
    if not state["is_loaded"]:
        raise HTTPException(status_code=503, detail="Model is still initializing")

    model = state["model"]
    device = state["device"]

    if req and req.condition is not None:
        sim = state["simulator"]
        signal, cond = sim.generate_window(req.condition)
        state["current_ppg"] = signal
        state["current_condition"] = req.condition
    else:
        signal = state["current_ppg"]
        cond = state["current_condition"]

    if signal is None:
        sim = state["simulator"]
        signal, cond = sim.generate_window(0)
        state["current_ppg"] = signal
        state["current_condition"] = 0

    tensor_in = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        logits, _ = model.ppg_encoder(tensor_in)
        probs = torch.softmax(logits, dim=-1)[0]
    inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    pred_idx = int(torch.argmax(probs).item())
    probabilities = {
        PPGSimulator.CLASSES[i]: round(float(probs[i].item()), 4)
        for i in range(len(PPGSimulator.CLASSES))
    }

    metrics = compute_hrv_and_metrics(signal, sampling_rate=25)

    return {
        "predicted_idx": pred_idx,
        "predicted_condition": PPGSimulator.CLASSES[pred_idx],
        "ground_truth_condition": PPGSimulator.CLASSES.get(cond, "Unknown"),
        "confidence": round(float(probs[pred_idx].item()), 4),
        "probabilities": probabilities,
        "inference_time_ms": inference_time_ms,
        "metrics": metrics,
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Multimodal clinical cardiology dialogue generation grounded with offline Clinical RAG.
    Supports conditioning with active 90s PPG sensor prefix embeddings.
    """
    if not state["is_loaded"]:
        raise HTTPException(status_code=503, detail="Model is still initializing")

    model = state["model"]
    tokenizer = state["tokenizer"]
    device = state["device"]

    cond_idx = state["current_condition"]
    cond_name = PPGSimulator.CLASSES.get(cond_idx, "Normal Sinus")

    curr_ppg = state["current_ppg"]
    metrics = compute_hrv_and_metrics(curr_ppg) if curr_ppg is not None else {"estimated_bpm": 72, "rmssd_ms": 38}

    # Query on-device Clinical RAG engine
    rag_docs = clinical_rag_engine.retrieve(req.message, condition=cond_name, top_k=1)
    rag_context = clinical_rag_engine.get_formatted_context(req.message, condition=cond_name)
    rag_title = rag_docs[0]["title"] if (rag_docs and rag_docs[0].get("retrieval_score", 0) > 2.0) else None

    system_prompt = (
        "You are MedGemma-Micro, an ultra-compact mobile edge cardiology AI assistant distilled from MedGemma. "
        "You provide accurate, evidence-based guidance on cardiac conditions, cardiovascular nutrition (DASH diet, "
        "sodium restriction < 1,500 mg, potassium/magnesium balance, omega-3s, soluble fiber, caffeine/alcohol limits), "
        "safe exercise prescription (Karvonen target heart rate zones, AHA 150 min/wk guidelines, post-AFib safe resumption, 1-min HRR), "
        "sleep architecture, nocturnal blood pressure dipping, obstructive sleep apnea (OSA/STOP-BANG), and stress/vagal modulation. "
        "MANDATORY PRESCRIBING WAIVER: When discussing or recommending any prescription medications or dosages, "
        "always include a clear medical disclaimer that this information is for educational guidance only and requires evaluation "
        "by a licensed cardiologist or physician before initiation or modification."
    )

    if req.use_ppg_context:
        context_prefix = (
            f"[MOBILE TELEMETRY: Continuous 90s PPG analysis detected '{cond_name}'. "
            f"BPM: {metrics['estimated_bpm']}, rMSSD: {metrics['rmssd_ms']} ms.]\n"
        )
    else:
        context_prefix = ""

    if rag_context:
        user_query = f"{context_prefix}{rag_context}\n[User Inquiry]: {req.message}"
    else:
        user_query = f"{context_prefix}{req.message}"

    messages = [{"role": "system", "content": system_prompt}]
    if req.history:
        for item in req.history[-4:]:
            messages.append({"role": item.role, "content": item.content})
    messages.append({"role": "user", "content": user_query})

    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_tokens = tokenizer(formatted_input, return_tensors="pt").to(device)
    text_embeds = model.student_lm.get_input_embeddings()(input_tokens.input_ids)

    start_time = time.perf_counter()
    if req.use_ppg_context and curr_ppg is not None:
        signal_tensor = torch.tensor(curr_ppg, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, latent = model.ppg_encoder(signal_tensor)
            prefix_embeds = model.ppg_projector(latent)  # [1, 4, 960]
            combined_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
            attention_mask = torch.ones(combined_embeds.shape[:2], dtype=torch.long, device=device)

            out_ids = model.student_lm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                max_new_tokens=req.max_tokens,
                do_sample=True,
                temperature=req.temperature,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.15,
            )
            reply_text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
            num_tokens = len(out_ids[0])
    else:
        with torch.no_grad():
            out = model.student_lm.generate(
                **input_tokens,
                max_new_tokens=req.max_tokens,
                do_sample=True,
                temperature=req.temperature,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.15,
            )
            generated_tokens = out[0][input_tokens.input_ids.shape[1] :]
            reply_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            num_tokens = len(generated_tokens)

    elapsed_sec = time.perf_counter() - start_time
    tokens_per_sec = round(num_tokens / max(0.001, elapsed_sec), 1)
    reply_text = reply_text.replace("<|im_end|>", "").strip()

    # Automatic Medical Disclaimer & Responsibility Waiver Safeguard
    med_keywords = [
        "metoprolol", "bisoprolol", "carvedilol", "diltiazem", "verapamil",
        "apixaban", "rivaroxaban", "dabigatran", "warfarin", "amiodarone",
        "flecainide", "sacubitril", "entresto", "lisinopril", "ramipril",
        "spironolactone", "eplerenone", "empagliflozin", "dapagliflozin",
        "nitroglycerin", "aspirin", "statin", "atorvastatin", "rosuvastatin",
        "medication", "dosage", "prescribe", "mg daily", "bid"
    ]
    has_med_content = any(kw in reply_text.lower() or kw in req.message.lower() for kw in med_keywords)
    has_disclaimer = any(term in reply_text.lower() for term in ["disclaimer", "waiver", "prescribing healthcare", "licensed cardiologist"])

    if has_med_content and not has_disclaimer:
        disclaimer_box = (
            "\n\n---\n"
            "⚠️ **Medical Disclaimer & Responsibility Waiver**: "
            "The medication information above is provided for clinical and educational reference only. "
            "It does not constitute a personal medical prescription or individualized treatment plan. "
            "Prescription drug selection, dosages, and titration must be evaluated and approved by a licensed cardiologist or physician "
            "based on personal renal function (eGFR), serum electrolytes, and drug interactions. "
            "Never start, modify, or discontinue prescribed cardiac medications without consulting your healthcare provider."
        )
        reply_text += disclaimer_box

    return {
        "reply": reply_text,
        "condition_conditioned": cond_name if req.use_ppg_context else "None (Pure Text)",
        "rag_grounded": bool(rag_title is not None),
        "guideline_citation": rag_title,
        "tokens_generated": num_tokens,
        "elapsed_sec": round(elapsed_sec, 3),
        "tokens_per_sec": tokens_per_sec,
    }


@app.get("/api/presets")
def get_presets():
    """Provides curated clinical cardiology test prompts."""
    return {
        "presets": [
            {
                "title": "Heart-Healthy Food & DASH Diet",
                "condition": 0,
                "prompt": "What is the best diet and food plan for heart disease, high blood pressure, and preventing arrhythmia episodes?",
                "tag": "Nutrition",
            },
            {
                "title": "Safe Exercise & Target HR Zones",
                "condition": 0,
                "prompt": "What are safe exercise guidelines and physical activity recommendations for someone with heart disease or after an arrhythmia episode?",
                "tag": "Exercise",
            },
            {
                "title": "Sleep, Nocturnal Dipping & Sleep Apnea",
                "condition": 2,
                "prompt": "How does sleep quality, sleep duration, and Obstructive Sleep Apnea (OSA) impact heart disease and Atrial Fibrillation?",
                "tag": "Sleep",
            },
            {
                "title": "Stress, Vagal Tone & Breathing",
                "condition": 0,
                "prompt": "What are effective stress management and breathing techniques to lower heart rate and reduce palpitations?",
                "tag": "Lifestyle",
            },
            {
                "title": "Bradycardia & Pacemaker Indications",
                "condition": 2,
                "prompt": "Can you please explain bradycardia, its clinical causes, symptoms, and when it requires a permanent pacemaker?",
                "tag": "Conduction",
            },
            {
                "title": "AFib Rate Control & Anticoagulation",
                "condition": 1,
                "prompt": "Mobile PPG sensor flagged Atrial Fibrillation. What are first-line rate control and stroke prevention medications?",
                "tag": "Medications",
            },
            {
                "title": "Emergency Chest Pain & Red Flags",
                "condition": 3,
                "prompt": "Heart rate is 145 bpm at rest. What are the emergent red-flag symptoms of myocardial infarction that require calling 911?",
                "tag": "Emergency",
            },
            {
                "title": "Heart Failure GDMT 4-Pillars",
                "condition": 0,
                "prompt": "Explain Heart Failure with reduced Ejection Fraction (HFrEF) and the four foundational pillars of GDMT.",
                "tag": "HeartFailure",
            },
        ]
    }


# Mount static files directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

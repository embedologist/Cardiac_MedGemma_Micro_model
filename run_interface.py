#!/usr/bin/env python3
"""
Launcher for MedGemma-Micro Interactive Test & Chat Interface
============================================================
Boots FastAPI / Uvicorn server and provides terminal access link.
"""

import os
import sys

# Auto-delegate to .venv/bin/python if running outside virtual environment
venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "bin", "python"))
if os.path.exists(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
    os.execv(venv_python, [venv_python] + sys.argv)

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    print("=" * 65)
    print(f"Starting MedGemma-Micro Interactive Test Interface")
    print(f"Model: medgemma_micro_cardio_edge.safetensors (Llama-3.2-1B Multimodal Edge)")
    print(f"URL: http://{host}:{port}")
    print("=" * 65)
    uvicorn.run("app:app", host=host, port=port, log_level="info", reload=False)


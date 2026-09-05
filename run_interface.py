#!/usr/bin/env python3
"""
Launcher for MedGemma-Micro Interactive Test & Chat Interface
============================================================
Boots FastAPI / Uvicorn server and provides terminal access link.
"""

import sys
import uvicorn

if __name__ == "__main__":
    port = 8000
    host = "127.0.0.1"
    print("=" * 65)
    print(f"Starting MedGemma-Micro Interactive Test Interface")
    print(f"URL: http://{host}:{port}")
    print("=" * 65)
    uvicorn.run("app:app", host=host, port=port, log_level="info", reload=False)

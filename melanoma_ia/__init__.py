"""
Single source of truth for:
    - model architecture and checkpoint loading: model.py
    - inference engine / preprocessing: inference.py
    - Grad-CAM: gradcam.py
    - environment configuration: config.py

Consumed by three different interfaces (CLI, HTTP API, research GUI), all importing from this package.
"""

__version__ = "1.0.0"

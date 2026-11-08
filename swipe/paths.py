"""Checkout-relative resources shared by the local application."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models'
WEB_DIR = ROOT / 'web'

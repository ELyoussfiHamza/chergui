"""Where things live on this machine. Override any of these with environment variables (or .env)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PX4_DIR = Path(os.getenv("PX4_DIR", Path.home() / "PX4-Autopilot")).expanduser()
PX4_BUILD = PX4_DIR / "build" / "px4_sitl_default"
PX4_BIN = PX4_BUILD / "bin" / "px4"
PX4_GZ_MODELS = PX4_DIR / "Tools" / "simulation" / "gz" / "models"
PX4_GZ_PLUGINS = PX4_BUILD / "src" / "modules" / "simulation" / "gz_plugins"
PX4_GZ_SERVER_CONFIG = PX4_DIR / "src" / "modules" / "simulation" / "gz_bridge" / "server.config"

WORLDS_DIR = PROJECT_ROOT / "worlds"
RUNS_DIR = Path(os.getenv("CHERGUI_RUNS_DIR", PROJECT_ROOT / "runs"))

# Gazebo's Python bindings are packaged for the system interpreter, not our venv.
SYSTEM_PYTHON = os.getenv("CHERGUI_SYSTEM_PYTHON", "/usr/bin/python3")

# MAVSDK listens here; PX4 instance 0 sends its offboard/API link to UDP 14540.
MAVLINK_URL = os.getenv("PX4_ADDRESS", "udpin://0.0.0.0:14540")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

import sys
from pathlib import Path

# Add repo root so `gpu_server` and `nano_client` are importable as packages.
sys.path.insert(0, str(Path(__file__).parent))

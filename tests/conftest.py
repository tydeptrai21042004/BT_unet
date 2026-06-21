from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep CPU-only CI/local smoke tests fast and deterministic. The benchmark
# scripts still use the user's normal PyTorch threading unless they run pytest.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

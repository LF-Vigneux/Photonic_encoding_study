import numpy as np
import torch.nn as nn
import torch
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


def classical_complexity(X: torch.Tensor, Y: torch.Tensor) -> float:
    pass


def induced_quantum_complxity(
    X: torch.Tensor, Y: torch.Tensor, encoding: nn.Module
) -> float:
    pass


def quantum_complexity(X: torch.Tensor, Y: torch.Tensor) -> float:
    pass

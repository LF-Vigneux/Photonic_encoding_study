"""Map legacy ``papers.nn_embedding`` imports to the local ``nn_embedding`` tree."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[2])]

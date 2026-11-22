"""Offline tests. Run from the repository root:

    python -m unittest discover -s tests -t . -v

They need only numpy and use a fake LLM and a hashing embedder, so no GPU, model or network is used.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

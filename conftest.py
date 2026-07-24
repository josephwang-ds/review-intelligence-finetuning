"""Root conftest — makes `src/` importable from both tests/ and eval/ without
repeating the sys.path hack in every test file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

"""Install pinned Jumper meshes without importing mjlab or changing dependencies."""

import argparse
import importlib.util
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, help="verified local upstream checkout")
    args = ap.parse_args()
    file = (
        Path(__file__).resolve().parents[1]
        / "src/mjlab_microduck/robot/jumper_model.py"
    )
    spec = importlib.util.spec_from_file_location("jumper_model", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.fetch(args.source)
    print(module.scene_xml())

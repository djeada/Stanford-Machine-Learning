"""Verify the executable examples and their inline copies in the notes."""

import ast
import re
import runpy
from pathlib import Path

import numpy as np
import scipy
import sklearn


def main():
    root = Path(__file__).resolve().parents[1]
    examples = sorted((root / "examples").glob("[0-9][0-9]_*.py"))
    assert len(examples) == 17, "Expected one example for each note chapter"
    print(
        f"NumPy {np.__version__}; SciPy {scipy.__version__}; "
        f"scikit-learn {sklearn.__version__}"
    )
    for example in examples:
        chapter = example.name[:2]
        notes = list((root / "slides").glob(f"{chapter}_*.md"))
        assert len(notes) == 1, f"Missing or ambiguous note for {example.name}"
        expected = ast.dump(ast.parse(example.read_text(encoding="utf-8")))
        fence = chr(96) * 3
        blocks = re.findall(
            fence + r"python\s*\n([\s\S]*?)" + fence,
            notes[0].read_text(encoding="utf-8"),
        )
        parsed = [ast.dump(ast.parse(block)) for block in blocks]
        assert expected in parsed, f"Inline example differs: {example.name}"
        print(f"\n[{chapter}] {example.name}")
        runpy.run_path(str(example), run_name="__main__")
    print(f"\nVerified all {len(examples)} examples and their inline copies.")


if __name__ == "__main__":
    main()

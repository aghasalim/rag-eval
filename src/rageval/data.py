import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "data/sklearn_docs/doc/modules"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_docs(root: Path = DOCS) -> dict[str, str]:
    """filename -> whitespace-normalised text. Whitespace is collapsed so evidence strings match across line breaks."""
    return {p.name: norm(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*.rst"))}


def chunk(docs: dict[str, str], size: int, overlap: int) -> list[tuple[str, str]]:
    """Fixed-size character windows per file (never crossing files). Returns (source, text)."""
    assert 0 <= overlap < size
    out = []
    for name, text in docs.items():
        for i in range(0, max(len(text) - overlap, 1), size - overlap):
            out.append((name, text[i : i + size]))
    return out

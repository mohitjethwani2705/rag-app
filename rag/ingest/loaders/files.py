import hashlib
from pathlib import Path

from .base import RawDoc

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def _source_id(path: Path) -> str:
    # Stable id from filename so re-ingesting a changed file replaces its chunks.
    return hashlib.sha1(path.name.encode()).hexdigest()[:16]


class FileLoader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[RawDoc]:
        ext = self.path.suffix.lower()
        if ext not in SUPPORTED:
            raise ValueError(f"Unsupported file type: {ext}")
        sid, title, uri = _source_id(self.path), self.path.name, str(self.path)

        if ext == ".pdf":
            import fitz  # PyMuPDF

            with fitz.open(self.path) as pdf:
                return [
                    RawDoc(p.get_text(), sid, title, uri, "pdf", {"page": i + 1})
                    for i, p in enumerate(pdf)
                    if p.get_text().strip()
                ]
        if ext == ".docx":
            import docx

            d = docx.Document(str(self.path))
            text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
            return [RawDoc(text, sid, title, uri, "docx")]

        text = self.path.read_text(encoding="utf-8", errors="replace")
        return [RawDoc(text, sid, title, uri, ext.lstrip("."))]

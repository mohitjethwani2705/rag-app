from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..core.config import settings
from .loaders.base import RawDoc


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def chunk_docs(docs: list[RawDoc]) -> list[Chunk]:
    # ~4 chars/token => chunk_size is in tokens, convert to characters.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size * 4,
        chunk_overlap=settings.chunk_overlap * 4,
        separators=["\n## ", "\n# ", "\n\n", "\n", ". ", " "],
    )
    chunks: list[Chunk] = []
    for doc in docs:
        for i, piece in enumerate(splitter.split_text(doc.text)):
            page = doc.metadata.get("page")
            cid = f"{doc.source_id}:{page or 0}:{i}"
            meta = {
                "source_id": doc.source_id,
                "title": doc.title,
                "uri": doc.uri,
                "source_type": doc.source_type,
                "page": page or 0,
            }
            chunks.append(Chunk(cid, piece, meta))
    return chunks

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RawDoc:
    text: str
    source_id: str
    title: str
    uri: str
    source_type: str
    metadata: dict = field(default_factory=dict)  # e.g. {"page": 3}


class Loader(Protocol):
    def load(self) -> list[RawDoc]: ...

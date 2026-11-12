"""Service contracts. Tests can supply fakes without loading torch or Gradio."""
from typing import Any, Iterator, Protocol


class RewriteService(Protocol):
    def validate(self, text: str, method: str) -> None: ...
    def stream(self, text: str, method: str) -> Iterator[str | dict[str, Any]]: ...


class DetectorService(Protocol):
    def explain(self, text: str) -> dict[str, Any]: ...


class ExportService(Protocol):
    def save(self, record: dict, edited: str, cache: dict | None) -> str: ...

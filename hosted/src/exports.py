"""User-controlled JSON downloads. No automatic draft history database."""
import json
import tempfile
import time
from pathlib import Path
from uuid import uuid4
from .config import SETTINGS
from .records import now


class JsonExportStore:
    def __init__(self, directory=None):
        self.directory = Path(directory or Path(tempfile.gettempdir())/'writing-desk-exports')
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, record, edited, cache):
        if not record or not record.get('rewrite'):
            raise ValueError('Rewrite something first.')
        if not isinstance(edited, str) or len(edited) > SETTINGS.detector_max_chars:
            raise ValueError('Use at most 30,000 characters for an edited export.')
        self.cleanup()
        detection = cache['data'] if cache and cache.get('text') == edited else None
        payload = {**record, 'edited_rewrite': edited,
                   'edited_ai_detection': detection, 'saved_at': now()}
        # UUID names avoid collisions and do not accept a client-supplied path.
        path = self.directory/f'rewrite-{uuid4().hex}.json'
        with path.open('x', encoding='utf-8') as target:
            json.dump(payload, target, indent=2, ensure_ascii=False)
        return str(path)

    def cleanup(self):
        cutoff = time.time() - SETTINGS.export_retention_hours*3600
        for path in self.directory.glob('rewrite-*.json'):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except FileNotFoundError:
                pass

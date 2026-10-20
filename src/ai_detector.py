"""Local experimental detector. Only load model files you trained yourself."""
from pathlib import Path
import threading
import re
import joblib

from .paths import ROOT

class AIDetector:
    def __init__(self):
        self.path = ROOT / 'models' / 'ai_detector.joblib'
        self.bundle = None
        self.lock = threading.Lock()

    def score(self, text):
        if not isinstance(text, str) or not text.strip():
            return {'available': False, 'message': 'No text to check.'}
        if len(text) > 30000:
            return {'available': False, 'message': 'Use 30,000 characters or fewer.'}
        if len(text.split()) < 50:
            return {'available': False, 'message': 'Use at least 50 words for this detector.'}
        with self.lock:
            if self.bundle is None:
                if not self.path.exists():
                    return {'available': False, 'message': 'Detector not trained yet.'}
                self.bundle = joblib.load(self.path)
            model = self.bundle['model']
            ai_index = list(model.classes_).index(1)
            value = float(model.predict_proba([text])[0, ai_index])
        return {'available': True, 'score': round(value * 100, 1),
                'model_version': self.bundle['version'],
                'note': 'Experimental classifier score, not proof of authorship or percent of AI-written words. Not independently calibrated.'}


    def explain(self, text):
        score = self.score(text)
        if not score.get('available'):
            return score
        model = self.bundle['model']
        # Preserve exact sentence substrings for selecting text in the browser.
        # Protect a few common abbreviation/decimal periods before splitting.
        masked = re.sub(r'(?<=\d)\.(?=\d)', '\u2024', text)
        masked = re.sub(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc)\.',
                        lambda m: m.group().replace('.', '\u2024'), masked, flags=re.I)
        matches = list(re.finditer(r'[^.!?\n]+(?:[.!?]+|$)', masked))
        spans = [(m.start(), m.end()) for m in matches if m.group().strip()]
        sentences = [text[start:end] for start, end in spans]
        # The current model learned passages, so isolated sentence scores are
        # explicitly experimental. Very short fragments are left unassessed.
        eligible = [i for i, sentence in enumerate(sentences) if len(sentence.split()) >= 8]
        ai_index = list(model.classes_).index(1)
        values = model.predict_proba([sentences[i] for i in eligible])[:, ai_index] if eligible else []
        scores = dict(zip(eligible, values))
        details = []
        for i, sentence in enumerate(sentences):
            value = scores.get(i)
            details.append({'text': sentence, 'start': spans[i][0], 'end': spans[i][1],
                            'score': round(float(value)*100, 1) if value is not None else None,
                            'assessed': value is not None,
                            'highlight': bool(value >= 0.5) if value is not None else False})
        # Attribute each whitespace-delimited word once, even if punctuation
        # splits a word into multiple spans (for example 'Hello.World').
        words = list(re.finditer(r'\S+', text))
        total_words = len(words)
        flagged_words = assessed_words = 0
        cursor = 0
        for word in words:
            while cursor < len(details) and word.start() >= details[cursor]['end']:
                cursor += 1
            if cursor < len(details) and details[cursor]['start'] <= word.start() < details[cursor]['end']:
                flagged_words += int(details[cursor]['highlight'])
                assessed_words += int(details[cursor]['assessed'])
        return {**score, 'sentences': details,
                'flagged_percent': round(100 * flagged_words / total_words, 1) if total_words else 0,
                'flagged_words': flagged_words, 'total_words': total_words,
                'unassessed_words': max(0, total_words-assessed_words),
                'flagged_count': sum(d['highlight'] for d in details),
                'assessed_count': len(eligible), 'sentence_count': len(details),
                'sentence_threshold': 0.5, 'sentence_method': 'isolated_sentence_experimental',
                'explanation': 'Percentage is the share of words in flagged sentences, not verified AI authorship. The passage classifier and cutoff have not been validated on individual sentences. Fragments under 8 words are unassessed, not classified as human.'}


def register_detector(app):
    """Add detection to the existing Flask response without changing Qwen."""
    from flask import request, jsonify, send_from_directory
    detector = AIDetector()

    def safe_score(text):
        try:
            return detector.explain(text)
        except Exception:
            app.logger.exception('Detector failed')
            return {'available': False, 'message': 'Detector unavailable. Check the terminal.'}

    @app.after_request
    def attach_scores(response):
        if request.path == '/api/rewrite' and response.status_code == 200 and response.is_json:
            data = response.get_json()
            if isinstance(data, dict) and isinstance(data.get('rewrite'), str):
                data['ai_detection'] = {'original': safe_score(data.get('original')),
                                        'rewrite': safe_score(data['rewrite'])}
                response.set_data(app.json.dumps(data))
        return response

    @app.post('/api/detect')
    def detect():
        if request.headers.get('Origin') not in (None, request.host_url.rstrip('/')):
            return jsonify(error='Use the same app page.'), 403
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get('text'), str):
            return jsonify(error='Send a JSON object with a text string.'), 400
        return jsonify(safe_score(body['text']))

    @app.post('/api/detect-details')
    def detect_details():
        if request.headers.get('Origin') not in (None, request.host_url.rstrip('/')):
            return jsonify(error='Use the same app page.'), 403
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get('text'), str):
            return jsonify(error='Send a JSON object with a text string.'), 400
        try:
            return jsonify(detector.explain(body['text']))
        except Exception:
            app.logger.exception('Detector explanation failed')
            return jsonify(error='Could not explain this detector. Check the terminal.'), 500

    @app.get('/detector-widget.js')
    def detector_widget():
        return send_from_directory(ROOT / 'web', 'detector-widget.js')

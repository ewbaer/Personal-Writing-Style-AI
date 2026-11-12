"""Fixed rewrite workflow with injectable services and explicit run records."""
from copy import deepcopy
import logging
from .contracts import RewriteService, DetectorService, ExportService
from .config import SETTINGS
from .validation import validate_text, validate_method
from .records import new_run, update_run, error_record

log = logging.getLogger(__name__)


class RewriteWorkflow:
    def __init__(self, rewriter: RewriteService, detector: DetectorService, exports: ExportService, profile=None):
        self.rewriter, self.detector, self.exports = rewriter, detector, exports
        self.profile = profile

    def check(self, text, record=None, label='edited'):
        """Only temporary CPU detector errors receive one retry. Never retry generation."""
        failure = None
        for attempt in range(SETTINGS.detector_retries+1):
            try:
                data = self.detector.explain(text)
                if record is not None:
                    record['retry_counts'][label] = attempt
                return data
            except (TimeoutError, ConnectionError) as error:
                failure = error
                if attempt < SETTINGS.detector_retries:
                    continue
                break
            except Exception as error:
                failure = error
                break
        if record is not None:
            record['retry_counts'][label] = attempt
            error_record(record, 'detection_'+label, failure)
        log.warning('Detector failed run_id=%s error_type=%s',
                    record['run_id'] if record else 'edited-check', type(failure).__name__)
        return {'available': False, 'message': 'Detector unavailable. You can check the text again.'}

    def run(self, text, method):
        text, method = validate_text(text), validate_method(method)
        record = new_run(text, method, self.profile)
        def event(kind):
            return {'kind': kind, 'record': deepcopy(record)}
        yield event('started')
        iterator = None
        try:
            self.rewriter.validate(text, method)
            record['completed_steps'].append('validation')
            update_run(record, 'waiting')
            yield event('waiting')
            iterator = iter(self.rewriter.stream(text, method))
            complete = False
            for item in iterator:
                if isinstance(item, str):
                    record['rewrite'] = item
                    update_run(record, 'generating')
                    yield event('partial')
                elif isinstance(item, dict) and item.get('rewrite'):
                    record.update(item)
                    complete = True
            if not complete:
                raise RuntimeError('The model returned no completed rewrite.')
            record['completed_steps'].append('generation')
            update_run(record, 'checking')
            yield event('checking')
            for label, passage in [('original', text), ('rewrite', record['rewrite'])]:
                record['ai_detection'][label] = self.check(passage, record, label)
            if record['ai_detection']['rewrite'].get('available'):
                record['completed_steps'].append('detection')
                update_run(record, 'completed')
            else:
                update_run(record, 'completed_without_detection')
            yield event('completed')
        except GeneratorExit:
            update_run(record, 'cancelled')
            raise
        except Exception as error:
            low = str(error).lower()
            if 'quota' in low or 'zerogpu' in low or 'gpu task aborted' in low:
                record['user_error'] = 'GPU time is unavailable right now. Sign in to Hugging Face or try again later.'
            elif 'out of memory' in low:
                record['user_error'] = 'The GPU ran out of memory. Try a shorter draft.'
            elif 'not loaded' in low:
                record['user_error'] = 'The rewrite model did not load. Check the Space logs.'
            if isinstance(error, ValueError):
                record['user_error'] = str(error)
            error_record(record, record['status'], error)
            update_run(record, 'failed')
            log.warning('Rewrite failed run_id=%s error_type=%s', record['run_id'], type(error).__name__)
            yield event('failed')
        finally:
            if iterator is not None and hasattr(iterator, 'close'):
                iterator.close()

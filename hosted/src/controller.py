"""Translate workflow events into Gradio updates. No model setup or file paths."""
from copy import deepcopy
import gradio as gr
from .presentation import (IDLE_SIGNALS, EDITED_SIGNALS, signals_line, signals_details,
                           build_highlights, show_editor, show_highlights, run_details)
from .records import update_run


class DeskController:
    def __init__(self, workflow):
        self.workflow = workflow

    def on_rewrite(self, text, method, want_highlight):
        stream = self.workflow.run(text, method)
        try:
            for event in stream:
                record = event['record']
                kind = event['kind']
                update = dict(rewritten=gr.update(value=record['rewrite'], visible=True),
                              highlights=gr.update(visible=False), details='',
                              result=record, cache=None, run=f"Run {record['run_id']}")
                if kind in ('started', 'waiting', 'partial'):
                    update.update(status='Writing…' if kind == 'partial' else 'Waiting for a GPU…',
                                  signals='Waiting for the rewrite…')
                elif kind == 'checking':
                    update.update(status='Checking writing signals…', signals='Checking writing signals…',
                                  run=run_details(record))
                elif kind == 'failed':
                    update.update(status=record.get('user_error') or f"Could not finish rewrite. Run {record['run_id']}. Try a shorter draft or check Space availability.",
                                  signals='Rewrite did not finish; no detection result.')
                else:
                    data = record['ai_detection']['rewrite']
                    editor, view = show_highlights(record['rewrite'], data) if want_highlight else show_editor()
                    warning = ' Output hit its length limit; check the ending.' if record.get('truncated') else ''
                    update.update(rewritten=gr.update(value=record['rewrite'], visible=editor['visible']),
                                  highlights=view, status='Revision ready.'+warning,
                                  signals=signals_line(data), details=signals_details(data),
                                  run=run_details(record)+f"  \nRun {record['run_id']}",
                                  cache={'text': record['rewrite'], 'data': data})
                yield update
        except ValueError as error:
            raise gr.Error(str(error)) from error
        finally:
            stream.close()

    def on_check(self, edited, want_highlight):
        if not (edited or '').strip():
            return IDLE_SIGNALS, '', gr.update(), None
        data = self.workflow.check(edited)
        view = gr.update(value=build_highlights(edited, data), visible=True) if want_highlight else gr.update()
        return signals_line(data), signals_details(data), view, {'text': edited, 'data': data}

    def on_toggle_highlight(self, active, edited, cache):
        if not active:
            return *show_editor(), gr.update(), gr.update()
        data = cache['data'] if cache and cache.get('text') == edited else self.workflow.check(edited)
        editor, view = show_highlights(edited, data)
        return editor, view, signals_line(data), signals_details(data)

    def on_edit(self, edited, cache):
        if cache and cache.get('text') == edited:
            return gr.update(), gr.update()
        return EDITED_SIGNALS if (edited or '').strip() else IDLE_SIGNALS, ''

    def on_clear(self):
        return ('', gr.update(value='', visible=True), gr.update(visible=False), False,
                'Paste a draft to begin.', IDLE_SIGNALS, '', 'No rewrite in this session.',
                None, None, gr.update(interactive=False))

    def on_cancel(self, record):
        record = deepcopy(record)
        if record is None:
            return 'Queued rewrite cancelled. You can start another draft.', None, gr.update(interactive=False)
        if record and record.get('status') not in ('completed', 'completed_without_detection', 'failed'):
            update_run(record, 'cancelled')
            return 'Rewrite cancelled. You can start another draft.', record, gr.update(interactive=bool(record.get('rewrite')))
        return gr.update(), gr.update(), gr.update()

    def on_export(self, result, edited, cache):
        if not result:
            raise gr.Error('Rewrite something first.')
        record = deepcopy(result)
        # Recover detection on the completed/edited text without another model call.
        if cache and cache.get('text') == edited:
            record['ai_detection']['rewrite'] = cache['data']
            if record.get('status') == 'completed_without_detection' and cache['data'].get('available'):
                update_run(record, 'completed')
                record['completed_steps'].append('detection')
        try:
            return self.workflow.exports.save(record, edited, cache)
        except ValueError as error:
            raise gr.Error(str(error)) from error

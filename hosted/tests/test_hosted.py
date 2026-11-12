"""Architecture and UI integration tests. No Qwen download or GPU required."""
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.workflow import RewriteWorkflow
from src.exports import JsonExportStore
from src.controller import DeskController
from src.ui import build_app
from src.ui_assets import THEME, CSS, THEME_JS
from src.validation import validate_text, validate_method
from src.prompts import load_examples, build_messages, PROMPT_VERSION
from src.ai_detector import AIDetector
from src.presentation import build_highlights, EDITED_SIGNALS

TEXT = ' '.join(['I worked on documentation and made the setup easier for our project team.']*8)


class FakeRewriter:
    def __init__(self, fail=False):
        self.calls = 0
        self.closed = False
        self.fail = fail

    def validate(self, text, method):
        validate_text(text)
        validate_method(method)

    def stream(self, text, method):
        self.calls += 1
        try:
            yield text[:20]
            if self.fail:
                raise RuntimeError('private error content')
            yield text
            yield dict(rewrite=text, model='fake', seconds=.01, gpu='CPU test',
                       first_token_seconds=.001, output_tokens=30, prompt_tokens=50,
                       precision='test', truncated=False, example_ids=[], max_new_tokens=1200)
        finally:
            self.closed = True


class FakeDetector:
    def __init__(self, errors=0, permanent=False):
        self.calls = 0
        self.errors = errors
        self.permanent = permanent

    def explain(self, text):
        self.calls += 1
        if self.permanent:
            raise ValueError('private detector content')
        if self.calls <= self.errors:
            raise TimeoutError('temporary')
        return dict(available=True, score=42., flagged_percent=25., flagged_count=1,
                    sentence_count=4, flagged_words=10, total_words=40,
                    unassessed_words=0, sentences=[])


@pytest.fixture
def services(tmp_path):
    rewriter, detector = FakeRewriter(), FakeDetector()
    workflow = RewriteWorkflow(rewriter, detector, JsonExportStore(tmp_path))
    return workflow, rewriter, detector


@pytest.mark.parametrize('text', ['', ' ', None, 7, 'x'*20001, 'word '*1001])
def test_invalid_input_never_calls_model(services, text):
    w, r, _ = services
    with pytest.raises(ValueError):
        list(w.run(text, 'basic'))
    assert r.calls == 0


def test_invalid_method(services):
    w,r,_ = services
    with pytest.raises(ValueError):list(w.run(TEXT, 'injected'))
    assert not r.calls


def test_run_record_and_snapshot_isolation(services):
    w,r,d = services
    events = list(w.run(TEXT, 'basic'))
    start, final = events[0]['record'], events[-1]['record']
    assert start['status'] == 'validating' and start['rewrite'] == ''
    assert final['status'] == 'completed' and final['rewrite'] == TEXT
    assert final['prompt_version'] == PROMPT_VERSION
    assert final['adapter'] is None and final['schema_version'] == 1
    assert final['completed_steps'] == ['validation','generation','detection']
    assert r.calls == 1 and d.calls == 2


def test_two_visitors_have_separate_records(services):
    w,_,_ = services
    with ThreadPoolExecutor(2) as pool:
        a,b = list(pool.map(lambda text:list(w.run(text, 'basic'))[-1]['record'],[TEXT, TEXT+' Second visitor.']))
    assert a['run_id'] != b['run_id'] and a['rewrite'] != b['rewrite']
    a['errors'].append({'test':True})
    assert not b['errors']


def test_temporary_detector_error_retries_once(services):
    w,r,_ = services
    w.detector = FakeDetector(errors=1)
    final=list(w.run(TEXT,'basic'))[-1]['record']
    assert final['status']=='completed' and final['retry_counts']['original']==1
    assert w.detector.calls==3 and r.calls==1


def test_detector_failure_keeps_output_and_can_recover(services):
    w,r,_=services;w.detector=FakeDetector(permanent=True)
    final=list(w.run(TEXT,'basic'))[-1]['record']
    assert final['status']=='completed_without_detection' and final['rewrite']==TEXT
    assert len(final['errors'])==2 and 'private' not in json.dumps(final['errors'])
    w.detector=FakeDetector()
    controller=DeskController(w)
    _,_,_,cache=controller.on_check(TEXT,False)
    path=controller.on_export(final,TEXT,cache)
    saved=json.loads(Path(path).read_text())
    assert saved['status']=='completed' and saved['edited_ai_detection']['available']
    assert r.calls==1


def test_temporary_retry_limit(services):
    w,_,_=services;w.detector=FakeDetector(errors=99)
    final=list(w.run(TEXT,'basic'))[-1]['record']
    assert w.detector.calls==4 and final['retry_counts']=={'original':1,'rewrite':1}


def test_generation_errors_stop_without_retry(services):
    w,_,_=services;w.rewriter=FakeRewriter(fail=True)
    final=list(w.run(TEXT,'basic'))[-1]['record']
    assert final['status']=='failed' and w.rewriter.calls==1
    assert not final['ai_detection'] and final['errors'][0]['type']=='RuntimeError'


def test_cancellation_closes_model_stream(services):
    w,r,_=services
    controller=DeskController(w);it=controller.on_rewrite(TEXT,'basic',False)
    next(it);next(it);partial=next(it)
    assert not r.closed
    it.close()
    assert r.closed
    _,record,_=controller.on_cancel(partial['result'])
    assert record['status']=='cancelled'
    assert partial['result']['status']=='generating'


def test_finished_run_is_not_marked_cancelled(services):
    w,_,_=services;c=DeskController(w)
    record=list(w.run(TEXT,'basic'))[-1]['record']
    assert c.on_cancel(record)[1].get('__type__')=='update'
    assert record['status']=='completed'


def test_exports_unique_and_stale_detection_not_reused(services):
    w,_,_=services;rec=list(w.run(TEXT,'basic'))[-1]['record']
    cache={'text':TEXT,'data':rec['ai_detection']['rewrite']}
    a=w.exports.save(rec,TEXT+' changed',cache);b=w.exports.save(rec,TEXT,cache)
    assert a!=b
    assert json.loads(Path(a).read_text())['edited_ai_detection'] is None
    assert json.loads(Path(b).read_text())['edited_ai_detection']['available']
    assert 'saved_at' in json.loads(Path(b).read_text())


def test_export_requires_rewrite(services):
    w,_,_=services
    with pytest.raises(ValueError):w.exports.save({},'',None)


def test_old_exports_cleaned_up(services):
    import os,time
    w,_,_=services;p=w.exports.directory/'rewrite-old.json';p.write_text('{}')
    os.utime(p,(time.time()-90000,time.time()-90000))
    keep=w.exports.directory/'unrelated.json';keep.write_text('{}')
    w.exports.cleanup()
    assert not p.exists() and keep.exists()


def test_real_detector_and_safe_highlights():
    result=AIDetector().explain(TEXT)
    assert result['available'] and 0<=result['score']<=100
    assert not AIDetector().score('short')['available']
    rendered=build_highlights('<script>alert(1)</script>',None)
    assert '<script>' not in rendered and '&lt;script&gt;' in rendered


def test_example_exclusion_and_cap():
    rows=load_examples();messages,ids=build_messages(rows[0]['AI Version'],'examples',rows)
    assert rows[0]['ID'] not in ids and len(ids)<=3
    with pytest.raises(ValueError):build_messages(TEXT,'examples',[])


def test_edit_invalidates_cached_score(services):
    c=DeskController(services[0])
    assert c.on_edit(TEXT+' changed',{'text':TEXT})[0]==EDITED_SIGNALS
    assert c.on_clear()[8] is None


def test_ui_build_and_http(services):
    import urllib.request
    demo=build_app(DeskController(services[0]))
    demo.queue(max_size=30,default_concurrency_limit=1)
    try:
        _,url,_=demo.launch(theme=THEME,css=CSS,js=THEME_JS,prevent_thread_lock=True,quiet=True,
                             server_name='127.0.0.1',server_port=7865)
        with urllib.request.urlopen(url+'config',timeout=15) as response:
            config=json.load(response)
        labels=[x['props'].get('value') for x in config['components']]
        assert 'Cancel' in labels and 'Rewrite ↗' in labels
        assert any(x.get('cancels') for x in config['dependencies'])
    finally:demo.close()


def test_cancel_before_run_starts(services):
    line, record, button = DeskController(services[0]).on_cancel(None)
    assert 'cancelled' in line and record is None and not button['interactive']

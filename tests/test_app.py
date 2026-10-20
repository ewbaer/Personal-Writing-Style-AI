"""Run with: python -m pytest -q. No Qwen download or GPU required.
Synthetic fixtures test plumbing, not detection quality.
"""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ai_detector import AIDetector
from src.rewrite_engine import RewriteEngine, build_messages, load_examples

AI = 'Comprehensive implementation delivers effective innovative solutions across strategic operational workflows.'
HUMAN = 'Yesterday I broke the old shelf and fixed it with spare wood.'
TEXT = ' '.join([AI, HUMAN, AI, HUMAN, AI, HUMAN])

@pytest.fixture
def detector(tmp_path):
    model = Pipeline([('words', TfidfVectorizer(ngram_range=(1,2))),
                      ('classifier', LogisticRegression(C=100))])
    model.fit([AI]*25+[HUMAN]*25,[1]*25+[0]*25)
    path=tmp_path/'detector.joblib'
    joblib.dump({'model':model,'version':'synthetic-test-only'},path)
    d=AIDetector();d.path=path
    return d

def test_missing_model_and_input_limits(tmp_path):
    d=AIDetector();d.path=tmp_path/'missing.joblib'
    assert not d.explain('')['available']
    assert '50 words' in d.explain('short text')['message']
    assert not d.explain('x '*16000)['available']
    assert 'not trained' in d.explain(TEXT)['message']

def test_coverage_and_offsets(detector):
    result=detector.explain(TEXT)
    assert result['available'] and result['sentence_count']==6
    assert result['flagged_count']==3
    assert result['flagged_words']==3*len(AI.split())
    assert result['flagged_percent']==round(100*result['flagged_words']/len(TEXT.split()),1)
    for s in result['sentences']:
        assert TEXT[s['start']:s['end']]==s['text']

def test_all_none_short_and_punctuation(detector):
    assert detector.explain(' '.join([AI]*6))['flagged_percent']==100
    assert detector.explain(' '.join([HUMAN]*6))['flagged_percent']==0
    r=detector.explain(TEXT+'\nHi. Yes! 3.14 Dr. Smith. '+AI.replace(' ','\t'))
    assert 0<=r['flagged_percent']<=100
    assert r['unassessed_words']>0
    assert r['flagged_words']<=r['total_words']
    for s in r['sentences']:
        if len(s['text'].split())<8: assert s['score'] is None and not s['highlight']

def test_prompting_uses_train_and_excludes_matching_input():
    rows=load_examples()
    assert len(rows)==6
    messages,ids=build_messages(rows[0]['AI Version'],'examples',rows)
    assert str(rows[0]['ID']) not in ids and len(ids)==3
    assert messages[-1]['content']==rows[0]['AI Version']
    assert build_messages('new input','basic',[])[1]==[]
    with pytest.raises(ValueError):build_messages('new input','bad',[])

def test_rewrite_validation_and_busy():
    e=RewriteEngine()
    for text,method in [(None,'basic'),('','basic'),('x '*1001,'basic'),('x'*20001,'basic'),('x','bad')]:
        with pytest.raises(ValueError):e.rewrite(text,method)
    e.lock.acquire()
    try:
        with pytest.raises(ValueError,match='already running'):e.rewrite('hello','basic')
    finally:e.lock.release()

@pytest.fixture
def client(monkeypatch,detector):
    import src.ai_detector as ai_detector
    monkeypatch.setattr(ai_detector,'AIDetector',lambda:detector)
    import app
    app=importlib.reload(app)
    monkeypatch.setattr(app.engine,'rewrite',lambda text,method:dict(original=text,rewrite=TEXT,method=method,model='TEST ONLY',seconds=.01,gpu='stub',example_ids=[],prompt_tokens=0,output_tokens=0,truncated=False))
    app.app.config.update(TESTING=True)
    return app.app.test_client()

def test_endpoints_and_response_attachment(client):
    assert client.get('/').status_code==200
    for path in ['/web/style.css','/web/app.js','/web/theme.js','/web/detector-widget.js','/detector-widget.js']:
        assert client.get(path).status_code==200
    response=client.post('/api/rewrite',json={'text':TEXT,'method':'basic'})
    assert response.status_code==200
    data=response.get_json()
    assert data['rewrite']==TEXT
    assert data['ai_detection']['rewrite']['flagged_count']==3
    for route in ['/api/detect','/api/detect-details']:
        assert client.post(route,json={'text':TEXT}).get_json()['flagged_count']==3
        assert client.post(route,json=[]).status_code==400
        assert client.post(route,json={'text':10}).status_code==400
        assert client.post(route,json={'text':TEXT},headers={'Origin':'https://other.example'}).status_code==403
    assert client.post('/api/rewrite',json=[],headers={}).status_code==400
    assert client.post('/api/rewrite',json={'text':TEXT},headers={'Origin':'https://other.example'}).status_code==403
    r=client.post('/api/rewrite',json={'text':'x'*100001})
    assert r.status_code==413 and r.is_json

def test_actual_training_pipeline(tmp_path,monkeypatch):
    import scripts.train_detector as train_detector
    # Unique group IDs and texts exercise the real grouped split, fitting,
    # evaluation, serialization and subsequent loading without the big corpus.
    rows=[]
    for label,base in [('ai',AI),('human',HUMAN)]:
        for i in range(120):rows.append(dict(text=(base+' ')*6+f' unique{i}',label=label,group_id=f'{label}{i}'))
    source=tmp_path/'data.csv';pd.DataFrame(rows).to_csv(source,index=False)
    monkeypatch.setattr(train_detector,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['train_detector.py','--csv',str(source)])
    train_detector.main()
    model=joblib.load(tmp_path/'models/ai_detector.joblib')['model']
    assert set(model.classes_)=={0,1}
    manifest=pd.read_csv(tmp_path/'results/detector/split_manifest.csv')
    assert set(manifest['split'])=={'train','validation','test'}
    assert manifest.groupby('group_id')['split'].nunique().max()==1

def test_prepare_public_csv(tmp_path):
    from scripts.prepare_detector_data import prepare
    rows=[]
    for label,base in [(0,HUMAN),(1,AI)]:
        for i in range(105):rows.append({'text':(base+' ')*6+f' unique{i}','generated':str(label)+'.0'})
    rows += [rows[0],{'text':'too short','generated':'unknown'}]
    path=tmp_path/'AI_Human.csv';pd.DataFrame(rows).to_csv(path,index=False)
    out=tmp_path/'prepared.csv';prepare(path,out,100)
    df=pd.read_csv(out)
    assert len(df)==200 and df['label'].value_counts().to_dict()=={'human':100,'ai':100}
    with pytest.raises(FileExistsError):prepare(path,out,100)

def fake_gpu_modules(monkeypatch,gb=16,free_gb=12,cuda=True):
    from contextlib import nullcontext
    captured={}
    class Tensor:
        def __init__(self,data):self.data=np.array(data)
        @property
        def shape(self):return self.data.shape
        def __getitem__(self,index):return Tensor(self.data[index])
        def __len__(self):return len(self.data)
        def item(self):return self.data.item()
    class Batch(dict):
        def to(self,device):captured['device']=device;return self
    class Tokenizer:
        eos_token_id=99
        def apply_chat_template(self,messages,**kwargs):
            captured['messages']=messages
            return Batch(input_ids=Tensor([[1,2,3]]))
        def decode(self,tokens,**kwargs):return 'A generated test rewrite.'
    class Model:
        device='cuda:0'
        generation_config=SimpleNamespace(eos_token_id=99)
        def eval(self):return self
        def generate(self,**kwargs):captured['generate']=kwargs;return Tensor([[1,2,3,7,99]])
    class Loader:
        @staticmethod
        def from_pretrained(model_id,**kwargs):captured['load']=kwargs;return Model()
    class TokenLoader:
        @staticmethod
        def from_pretrained(model_id):return Tokenizer()
    class BitsConfig:
        def __init__(self,**kwargs):captured['quantization']=kwargs
    torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:cuda,mem_get_info=lambda _: (free_gb*1024**3,gb*1024**3),is_bf16_supported=lambda:True,synchronize=lambda:None,get_device_name=lambda _: 'TEST GPU'),bfloat16='torch.bfloat16',float16='torch.float16',inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=TokenLoader,AutoModelForCausalLM=Loader,BitsAndBytesConfig=BitsConfig))
    monkeypatch.setitem(sys.modules,'bitsandbytes',SimpleNamespace())
    return captured

def test_generation_path_full_weights_and_reuse(monkeypatch):
    captured=fake_gpu_modules(monkeypatch)
    e=RewriteEngine();result=e.rewrite('A test input.','basic')
    assert result['rewrite']=='A generated test rewrite.' and result['precision']=='bfloat16'
    assert result['output_tokens']==2 and result['prompt_tokens']==3
    assert not result['truncated'] and not e.lock.locked()
    assert captured['load']['device_map']=={'':0}
    assert 'quantization_config' not in captured['load']
    model=e.model;e.rewrite('A second input.','examples');assert e.model is model

def test_small_gpu_quantization_and_failure_cleanup(monkeypatch):
    captured=fake_gpu_modules(monkeypatch,gb=6,free_gb=5)
    e=RewriteEngine();assert e.rewrite('A test.','basic')['precision']=='4bit'
    assert captured['quantization']['load_in_4bit'] is True
    fake_gpu_modules(monkeypatch,cuda=False)
    e=RewriteEngine()
    with pytest.raises(RuntimeError,match='No CUDA'):e.rewrite('A test.','basic')
    assert not e.lock.locked()
    fake_gpu_modules(monkeypatch,gb=16,free_gb=2)
    with pytest.raises(RuntimeError,match='Not enough free'):RewriteEngine().load()

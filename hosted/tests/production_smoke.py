import os,sys,json
from pathlib import Path
if len(sys.argv) != 2 or not Path(sys.argv[1]).is_dir():
    raise SystemExit('Usage: python tests/production_smoke.py <local-tiny-model-directory>')
os.environ['WRITING_MODEL_ID']=str(Path(sys.argv[1]).resolve())
os.environ['WRITING_MAX_NEW_TOKENS']='24'
os.environ['GRADIO_ANALYTICS_ENABLED']='False'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
torch.set_num_threads(2)
import app
from gradio_client import Client
print('production imports passed',flush=True)
app.demo.queue(max_size=30,default_concurrency_limit=1)
try:
 _,url,_=app.demo.launch(theme=app.THEME,css=app.CSS,js=app.THEME_JS,prevent_thread_lock=True,quiet=True,server_port=7866)
 print('launched',flush=True)
 c=Client(url,verbose=False)
 r=c.predict('This is a short note about documentation and writing better setup instructions.','basic',False,api_name='/rewrite_adapter')
 print('result',str(r)[:1500],flush=True)
 assert r[1].startswith('Revision ready'),r
 assert r[0]['value'].strip() if isinstance(r[0],dict) else r[0].strip()
 from concurrent.futures import ThreadPoolExecutor
 def another(i):
  visitor=Client(url,verbose=False)
  return visitor.predict(f'A separate visitor draft number {i} about project documentation.', 'basic', False, api_name='/rewrite_adapter')
 with ThreadPoolExecutor(2) as pool:
  results=list(pool.map(another,[1,2]))
 assert all(x[1].startswith('Revision ready') for x in results)
 # Confirm transformer generator cleanup releases its generation thread.
 from src.hosted_engine import HostedEngine
 import threading
 e=HostedEngine();e.load()
 _,tokens,_,_=e.prepare('A short draft about documentation.','basic')
 before={t.ident for t in threading.enumerate()}
 stream=e.stream(tokens);next(stream);stream.close()
 assert not [t for t in threading.enumerate() if t.ident not in before and t.is_alive()]
 print('two independent clients queued successfully; transformer generator cancellation cleaned up its worker',flush=True)
 print('real transformer streaming through ZeroGPU decorator (CPU passthrough) and Gradio queue passed',flush=True)
finally: app.demo.close()

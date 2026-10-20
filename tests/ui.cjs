// Run: npm install && npm test. Checks DOM behavior with a mocked model API.
const {JSDOM}=require('jsdom');
const fs=require('node:fs');const path=require('node:path');const assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.join(root,'web/index.html'),'utf8');
const dom=new JSDOM(html,{url:'http://127.0.0.1:7860',runScripts:'outside-only',pretendToBeVisual:true});
const w=dom.window,d=w.document;
w.ResizeObserver=class {observe(){}};
w.matchMedia=()=>({matches:true});
w.HTMLDialogElement.prototype.showModal=function(){this.open=true};
w.HTMLDialogElement.prototype.close=function(){this.open=false};
const tick=()=>new Promise(r=>setTimeout(r,0));
const output=d.getElementById('rewritten');
const text='This is the first sentence with enough words to check.\n\nThis is the second sentence with enough words to check.';
const first=text.split('\n')[0],second=text.split('\n').at(-1);
function signals(t){
 const sentences=t===text?[{text:first,highlight:true},{text:second,highlight:false}]:[{text:t,highlight:true}];
 return {available:true,score:77,flagged_percent:t===text?50:100,flagged_count:1,sentence_count:sentences.length,flagged_words:10,total_words:20,unassessed_words:0,sentences};
}
let failure=false,pending=null;
w.fetch=async(url,opts)=>{
 const body=JSON.parse(opts.body);
 if(failure)throw new Error('Test network error');
 if(pending)return pending;
 const data=url==='/api/rewrite'?{original:body.text,rewrite:text,model:'TEST ONLY',seconds:.1,gpu:'stub',example_ids:['1'],prompt_tokens:2,output_tokens:20,truncated:false,ai_detection:{rewrite:signals(text)}}:signals(body.text);
 return {ok:true,headers:{get:()=> 'application/json'},json:async()=>data};
};
const saved=[];
w.Blob=class{constructor(parts){saved.push(JSON.parse(parts.join('')))}};
w.URL.createObjectURL=()=> 'blob:test';w.URL.revokeObjectURL=()=>{};
w.HTMLAnchorElement.prototype.click=function(){};
Object.defineProperty(w.navigator,'clipboard',{value:{writeText:async text=>{w.copied=text}}});
for(const file of ['theme.js','detector-widget.js','app.js'])w.eval(fs.readFileSync(path.join(root,'web',file),'utf8'));
async function main(){
 const panel=d.getElementById('detectorPanel');const [check,toggle]=panel.querySelectorAll('button');
 assert(check.disabled && toggle.disabled);
 assert(panel.querySelector('.signal-status').textContent.includes('Rewrite text to see'));
 assert.equal(output.closest('.page').querySelectorAll('textarea').length,1);
 assert.equal(d.querySelectorAll('.revision-info').length,0);
 assert.equal(output.closest('.page').contains(panel),false);
 assert(output.closest('.rewritten-panel').querySelector('.panel-foot').compareDocumentPosition(panel)&w.Node.DOCUMENT_POSITION_FOLLOWING);
 d.getElementById('original').value='A test input';d.getElementById('rewrite').click();await tick();await tick();
 assert.equal(output.value,text);
 assert(panel.querySelector('.signal-status').textContent.includes('50.0%'));
 assert.equal(check.disabled,false);assert.equal(d.getElementById('outputCount').textContent,'20 words');
 toggle.click();await tick();
 assert.equal(toggle.getAttribute('aria-pressed'),'true');
 assert.equal(d.querySelector('.signal-overlay').hidden,false);
 assert.equal(d.querySelectorAll('.signal-overlay mark').length,1);
 assert.equal(d.querySelector('.signal-overlay mark').textContent,first);
 assert.equal(output.value,text);
 toggle.click();assert.equal(d.querySelector('.signal-overlay').hidden,true);assert.equal(toggle.getAttribute('aria-pressed'),'false');
 toggle.click();output.value=text+' An edit.';output.dispatchEvent(new w.Event('input'));
 assert.equal(d.querySelector('.signal-overlay').hidden,true);
 assert.equal(w.writingSignals.snapshot(),null);
 check.click();await tick();assert(panel.querySelector('.signal-status').textContent.includes('100.0%'));
 toggle.click();assert.equal(toggle.getAttribute('aria-pressed'),'true');
 d.getElementById('copy').click();await tick();assert.equal(w.copied,output.value);
 d.getElementById('save').click();assert.equal(saved.at(-1).edited_rewrite,output.value);assert.equal(saved.at(-1).edited_ai_detection.flagged_percent,100);
 const edited=output.value;d.getElementById('themeToggle').click();assert.equal(d.documentElement.dataset.theme,'light');assert.equal(output.value,edited);
 d.getElementById('themeToggle').click();assert.equal(d.documentElement.dataset.theme,'dark');
 d.getElementById('clear').click();await tick();assert.equal(output.value,'');assert(check.disabled&&toggle.disabled);
 // Late results must not apply to a newly edited draft.
 output.value=text;output.dispatchEvent(new w.Event('input'));let resolve;
 pending=new Promise(r=>resolve=r);check.click();output.value='Changed while checking.';output.dispatchEvent(new w.Event('input'));
 resolve({ok:true,json:async()=>signals(text)});pending=null;await tick();assert.equal(w.writingSignals.snapshot(),null);
 failure=true;check.click();await tick();assert(panel.querySelector('.signal-status').textContent.includes('Test network error'));
 d.getElementById('rewrite').click();await tick();await tick();assert.equal(d.getElementById('rewrite').disabled,false);
 assert.equal(w.writingSignals.snapshot(),null);
 console.log('UI checks passed: rewrite, count, signal status, inline toggle, editing, copy/export, themes, clear, stale responses and errors.');
 w.close();
}
main().catch(error=>{console.error(error);w.close();process.exitCode=1});

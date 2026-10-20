// Local Writing desk integration. Load before app.js.
(() => {
  const output = document.getElementById('rewritten');
  if (!output || document.getElementById('detectorPanel')) return;
  const make = (tag, cls, text) => {const el=document.createElement(tag);el.className=cls;if(text)el.textContent=text;return el;};
  const style=make('style','');
  style.textContent=`
  #detectorPanel{margin:6px 0 20px;padding:0 12px;border:0;background:transparent;color:var(--ink);font:12px/1.6 "Segoe UI",sans-serif}
  #detectorPanel h3{font:20px Georgia,serif;margin:0 0 8px}
  #detectorPanel .signal-status{font-variant-numeric:tabular-nums}
  #detectorPanel .signal-toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 10px}
  #detectorPanel button{min-height:36px;display:inline-flex;align-items:center;justify-content:center;background:var(--button-surface);color:var(--ink);border:1px solid transparent;border-radius:5px;padding:9px 14px;font:12px "Segoe UI",sans-serif}
  #detectorPanel button:hover:not(:disabled){background:var(--button-hover)}
  #detectorPanel button[aria-pressed="true"]{background:var(--ink);color:var(--paper)}
  #detectorPanel button:disabled{opacity:.4;cursor:default}
  #detectorPanel small,#detectorPanel details{display:block;color:var(--muted);font-size:11px}
  #detectorPanel details{margin-top:8px}#detectorPanel summary{cursor:pointer}
  .revision-helper{font:11px/1.5 "Segoe UI",sans-serif;color:var(--muted);margin:0 0 12px}
  .signal-overlay{position:absolute;overflow:hidden;pointer-events:none;z-index:0;color:transparent!important}
  .signal-overlay-inner{color:transparent!important;white-space:pre-wrap;overflow-wrap:anywhere}
  .signal-overlay mark{color:transparent!important;background:#e7d6aa;border-radius:2px;padding:0}
  [data-theme="dark"] .signal-overlay mark{background:#554831}
  @media(max-width:700px){#detectorPanel .signal-toolbar{flex-direction:column}#detectorPanel button{width:100%}}
  `;
  document.head.append(style);
  const panel=make('section','');panel.id='detectorPanel';panel.setAttribute('aria-label','Writing signals');
  const line=make('div','signal-status','Rewrite text to see writing signals.');line.setAttribute('role','status');
  const check=make('button','','Check edited output');check.type='button';
  const toggle=make('button','','Highlight sentences');toggle.type='button';toggle.setAttribute('aria-pressed','false');
  const toolbar=make('div','signal-toolbar');toolbar.append(check,toggle);
  const details=make('details','');details.append(make('summary','','Score details'));
  const detailText=make('div','');details.append(detailText);
  panel.append(make('h3','','Writing signals'),line,toolbar,details);
  const section=output.closest('.rewritten-panel') || output.closest('section');
  section?.querySelector('.revision-info')?.remove();
  const foot=section?.querySelector('.panel-foot');
  const helper=make('p','revision-helper','Review and edit anything that doesn’t sound like you');
  if(foot){foot.after(helper);helper.after(panel);}else{output.parentElement.after(helper);helper.after(panel);}

  // A non-editable background layer mirrors the textarea. The textarea remains
  // the only editor, preserving selection, undo, paste, copy, and saved values.
  const page=output.parentElement;page.style.position='relative';
  const overlay=make('div','signal-overlay');overlay.setAttribute('aria-hidden','true');overlay.hidden=true;
  const inner=make('div','signal-overlay-inner');overlay.append(inner);page.insertBefore(overlay,output);
  output.style.position='relative';output.style.zIndex='1';
  function geometry(){
    const cs=getComputedStyle(output);
    overlay.style.left=output.offsetLeft+'px';overlay.style.top=output.offsetTop+'px';
    overlay.style.width=output.clientWidth+'px';overlay.style.height=output.clientHeight+'px';
    for(const prop of ['fontFamily','fontSize','fontWeight','fontStyle','lineHeight','letterSpacing','wordSpacing','textAlign','textIndent','textTransform','tabSize','paddingTop','paddingRight','paddingBottom','paddingLeft','boxSizing','whiteSpace','overflowWrap','wordBreak'])inner.style[prop]=cs[prop];
    inner.style.width=output.clientWidth+'px';
    inner.style.transform=`translate(${-output.scrollLeft}px, ${-output.scrollTop}px)`;
  }
  output.addEventListener('scroll',geometry);window.addEventListener('resize',geometry);
  if(window.ResizeObserver)new ResizeObserver(geometry).observe(output);
  const original=document.getElementById('original');
  if(original && window.ResizeObserver){
    const sizes=new WeakMap();const obs=new ResizeObserver(entries=>{
      if(!window.matchMedia('(min-width:601px)').matches)return;
      for(const entry of entries){const height=entry.target.getBoundingClientRect().height;if(sizes.get(entry.target)===height)continue;
        sizes.set(entry.target,height);const other=entry.target===output?original:output;
        if(Math.abs(other.getBoundingClientRect().height-height)>1){other.style.height=height+'px';sizes.set(other,height);}}
    });obs.observe(original);obs.observe(output);
  }
  let revision=0,busy=false,cached=null,cachedText='',active=false;
  const fetchOriginal=window.fetch.bind(window);
  function sync(){check.disabled=toggle.disabled=busy || output.disabled || !output.value.trim();}
  function hide(){active=false;overlay.hidden=true;toggle.setAttribute('aria-pressed','false');}
  function invalidate(){revision++;cached=null;cachedText='';hide();inner.replaceChildren();detailText.textContent='';line.textContent=output.value.trim()?'Output edited · check again to update.':'Rewrite text to see writing signals.';sync();}
  function display(data){
    if(!data?.available){line.textContent=data?.message || 'Detector unavailable.';detailText.textContent='';return;}
    if(typeof data.flagged_percent!=='number'){line.textContent='Restart the app to load the updated detector.';return;}
    line.textContent=`Text flagged as AI-like: ${data.flagged_percent.toFixed(1)}% · ${data.flagged_count} of ${data.sentence_count} sentences`;
    detailText.textContent=`Whole-text classifier score: ${data.score.toFixed(1)} / 100. Flagged words: ${data.flagged_words} of ${data.total_words}. Unassessed words: ${data.unassessed_words}. Sentences under 8 words are left unassessed. Sentence cutoff: 50 / 100. The percentage measures words in flagged sentences, not verified AI authorship. Sentence scoring has not been validated.`;
  }
  function show(){
    if(!cached?.available || cachedText!==output.value)return;
    inner.replaceChildren();let cursor=0;
    for(const sentence of cached.sentences || []){
      const start=output.value.indexOf(sentence.text,cursor);if(start<0)continue;
      inner.append(document.createTextNode(output.value.slice(cursor,start)));
      inner.append(sentence.highlight?make('mark','',sentence.text):document.createTextNode(sentence.text));
      cursor=start+sentence.text.length;
    }
    inner.append(document.createTextNode(output.value.slice(cursor)+'\n'));
    active=true;overlay.hidden=false;toggle.setAttribute('aria-pressed','true');geometry();
  }
  async function analyze(wantHighlight=false){
    if(busy || output.disabled || !output.value.trim())return;
    const text=output.value,current=++revision;busy=true;sync();hide();line.textContent='Checking writing signals…';
    try{const response=await fetchOriginal('/api/detect-details',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
      const data=await response.json();if(current!==revision || text!==output.value)return;
      if(!response.ok)throw new Error(data.error || 'Detection failed.');cached=data;cachedText=text;display(data);if(wantHighlight)show();
    }catch(error){if(current===revision)line.textContent=error.message || 'Could not reach detector.';}
    finally{busy=false;sync();}
  }
  check.addEventListener('click',()=>analyze(active));
  toggle.addEventListener('click',()=>{if(active){hide();return;}if(cached && cachedText===output.value)show();else analyze(true);});
  output.addEventListener('input',invalidate);
  document.getElementById('clear')?.addEventListener('click',()=>queueMicrotask(invalidate));
  new MutationObserver(sync).observe(output,{attributes:true,attributeFilter:['disabled']});
  // Explicit app events keep rendering and checking independent of fetch.
  window.writingSignals={
    reset(message='Rewrite text to see writing signals.'){
      invalidate();line.textContent=message;
    },
    accept(data){
      revision++;hide();cached=data.ai_detection?.rewrite || null;
      cachedText=data.rewrite;display(cached);sync();
    },
    snapshot(){return cachedText===output.value?cached:null;}
  };
  sync();geometry();
})();

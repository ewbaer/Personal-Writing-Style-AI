const $ = id => document.getElementById(id);
let method = 'examples', result = null, busy = false;
const count = text => text.trim() ? text.trim().split(/\s+/).length : 0;
function counts() {
  $('inputCount').textContent = `${count($('original').value)} words`;
  $('outputCount').textContent = `${count($('rewritten').value)} words`;
  $('copy').disabled = busy || !$('rewritten').value.trim();
}
function status(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
}
function setBusy(value) {
  busy = value;
  for (const id of ['original','rewritten','rewrite','clear','voice']) $(id).disabled = value;
  $('save').disabled = value || !result;
  $('rewrite').textContent = value ? 'Rewriting…' : 'Rewrite ↗';
  counts();
}
$('voice').addEventListener('change', () => {
  method = $('voice').value;
  $('methodHint').textContent = method === 'examples'
    ? 'Uses up to 3 examples from your writing.'
    : 'Rewrites with general instructions, without examples.';
});
$('settingsOpen').addEventListener('click', () => $('settings').showModal());
$('settingsClose').addEventListener('click', () => $('settings').close());
$('settings').addEventListener('click', event => {
  const box = $('settings').getBoundingClientRect();
  if (event.target === $('settings') &&
      (event.clientX < box.left || event.clientX > box.right ||
       event.clientY < box.top || event.clientY > box.bottom)) $('settings').close();
});
document.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !busy && !$('settings').open) {
    event.preventDefault(); $('rewrite').click();
  }
});
$('original').addEventListener('input', counts);
$('rewritten').addEventListener('input', counts);
$('clear').addEventListener('click', () => {
  $('original').value = $('rewritten').value = '';
  result = null; $('save').disabled = true; counts(); $('runDetails').textContent = 'No rewrite in this session.'; status('Paste a draft to begin.'); $('original').focus();
});
$('copy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('rewritten').value); status('Rewrite copied.'); }
  catch { status('Select the rewrite and press Ctrl+C to copy.', true); }
});
$('rewrite').addEventListener('click', async () => {
  const text = $('original').value.trim();
  if (!text) return status('Paste some text first.', true);
  if (count(text) > 1000) return status('Use 1,000 words or fewer.', true);
  result = null; $('rewritten').value = ''; $('runDetails').textContent = 'Rewrite in progress.'; setBusy(true);
  status('Preparing your revision… First use can take a few minutes.');
  try {
    const response = await fetch('/api/rewrite', {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({text, method})
    });
    if (!response.headers.get('content-type')?.includes('application/json')) throw new Error('Server error. Check the app terminal.');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Rewrite failed.');
    result = data; $('rewritten').value = data.rewrite;
    const warning = data.truncated ? ' Output hit its length limit; check the ending.' : '';
    $('runDetails').textContent = `${data.model}\n${data.gpu} · ${data.seconds}s generation\n${data.example_ids.length} examples · ${data.prompt_tokens} input tokens · ${data.output_tokens} output tokens`;
    status(`Revision ready.${warning}`, data.truncated);
  } catch (error) { $('runDetails').textContent = 'The last rewrite did not finish.'; status(error.message || 'Could not reach the app. Check its terminal.', true); }
  finally { setBusy(false); }
});
$('save').addEventListener('click', () => {
  if (!result) return;
  // Keep generated text separate from edits; edits do not automatically retrain the model.
  const saved = {...result, edited_rewrite: $('rewritten').value, saved_at: new Date().toISOString()};
  const url = URL.createObjectURL(new Blob([JSON.stringify(saved, null, 2)], {type:'application/json'}));
  const a = document.createElement('a'); a.href = url; a.download = `rewrite-${Date.now()}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  status('Result downloaded, including your edits and the original model output.');
});

// Change appearance without reloading or touching the current draft/revision.
function updateThemeButton() {
  const dark = document.documentElement.dataset.theme === 'dark';
  const label = dark ? 'Switch to light mode' : 'Switch to dark mode';
  $('themeToggle').setAttribute('aria-label', label);
  $('themeToggle').title = label;
}
$('themeToggle').addEventListener('click', () => {
  const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('writing-desk-theme', theme); } catch (_) {}
  updateThemeButton();
});
updateThemeButton();

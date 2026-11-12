
() => {
  const apply = (dark) => {
    document.body.classList.toggle('dark', dark);
    document.documentElement.classList.toggle('dark', dark);
    document.querySelectorAll('gradio-app, .gradio-container').forEach(el => el.classList.toggle('dark', dark));
    document.documentElement.style.colorScheme = dark ? 'dark' : 'light';
  };
  let saved = null;
  try { saved = localStorage.getItem('writing-desk-theme'); } catch (_) {}
  if (saved === 'dark' || saved === 'light') apply(saved === 'dark');
  else if (!window.location.search.includes('__theme')) apply(true);
  window.writingDeskToggle = () => {
    const dark = !document.body.classList.contains('dark');
    apply(dark);
    try { localStorage.setItem('writing-desk-theme', dark ? 'dark' : 'light'); } catch (_) {}
  };
  window.writingDeskSettings = () => {
    const acc = document.querySelector('#score-details button, #score-details .label-wrap');
    if (acc) { acc.click(); acc.scrollIntoView({behavior: 'smooth', block: 'center'}); }
  };
  document.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      const button = document.querySelector('#rewrite-button button, button#rewrite-button');
      if (button && !button.disabled) { event.preventDefault(); button.click(); }
    }
  });
}

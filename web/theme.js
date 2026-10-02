// Apply the saved theme before styles load to avoid a bright flash on opening.
(() => {
  let theme = 'dark';
  try {
    const saved = localStorage.getItem('writing-desk-theme');
    if (saved === 'light' || saved === 'dark') theme = saved;
  } catch (_) { /* The toggle still works when browser storage is blocked. */ }
  document.documentElement.dataset.theme = theme;
})();

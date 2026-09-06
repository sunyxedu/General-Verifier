// Navigation uses native controls; the document also works without JavaScript.
const menuControl = document.querySelector('#menu-control');
const tocControl = document.querySelector('#toc-control');
const mobile = window.matchMedia('(max-width: 57rem)');
function closeMenus() {
  menuControl.checked = false;
  tocControl.checked = false;
}
mobile.addEventListener('change', closeMenus);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') closeMenus();
});
document.querySelector('.book-page').addEventListener('click', event => {
  if (!event.target.closest('.menu-label')) menuControl.checked = false;
});
document.querySelectorAll('.book-header nav a, .book-menu a').forEach(link => {
  link.addEventListener('click', closeMenus);
});

const sectionLinks = [...document.querySelectorAll('.book-toc a, .mobile-toc a')];
const headings = [...document.querySelectorAll('.markdown h2, .markdown h3')];
let scheduled = false;
function markSection() {
  const heading = headings.findLast(node => node.getBoundingClientRect().top <= 100);
  for (const link of sectionLinks) {
    if (heading && decodeURIComponent(link.hash.slice(1)) === heading.id) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  }
  scheduled = false;
}
document.addEventListener('scroll', () => {
  if (!scheduled) {
    scheduled = true;
    requestAnimationFrame(markSection);
  }
}, { passive: true });
window.addEventListener('hashchange', markSection);
markSection();

// A small local index keeps search usable offline and needs no external service.
const searchIndex = JSON.parse(document.querySelector('#search-index').textContent);
const search = document.querySelector('#book-search-input');
const results = document.querySelector('#book-search-results');
const status = document.querySelector('#search-status');
document.querySelector('.book-search').hidden = false;
function searchTerms(text) { return text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) || []; }
const indexed = searchIndex.map(entry => ({ ...entry, words: new Set(searchTerms(entry.title + ' ' + entry.page + ' ' + entry.text)) }));
search.addEventListener('input', () => {
  const terms = [...new Set(searchTerms(search.value))];
  results.replaceChildren();
  results.hidden = terms.length === 0;
  status.textContent = '';
  if (!terms.length) return;
  const matches = indexed.filter(entry => terms.every(term => [...entry.words].some(word => word.includes(term))))
    .map(entry => ({ ...entry, rank: terms.filter(term => entry.title.toLowerCase().includes(term)).length }))
    .sort((a, b) => b.rank - a.rank).slice(0, 8);
  for (const match of matches) {
    const item = document.createElement('li');
    const link = document.createElement('a');
    link.href = match.url;
    link.textContent = match.title;
    link.addEventListener('click', closeMenus);
    const detail = document.createElement('small');
    detail.textContent = match.page;
    item.append(link, detail);
    results.append(item);
  }
  const message = matches.length ? `${matches.length} results` : 'No matching sections';
  status.textContent = message;
  if (!matches.length) {
    const item = document.createElement('li');
    item.textContent = message;
    results.append(item);
  }
});
search.addEventListener('keydown', event => {
  if (event.key === 'ArrowDown') { event.preventDefault(); results.querySelector('a')?.focus(); }
  if (event.key === 'Escape') { search.value = ''; search.dispatchEvent(new Event('input')); }
});

if (navigator.clipboard && window.isSecureContext) {
  document.querySelectorAll('.markdown pre').forEach(pre => {
    const wrapper = document.createElement('div');
    wrapper.className = 'code-block';
    pre.before(wrapper);
    wrapper.append(pre);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-code';
    button.textContent = 'Copy';
    button.setAttribute('aria-label', 'Copy code');
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(pre.querySelector('code').textContent);
        button.textContent = 'Copied';
      } catch {
        button.textContent = 'Select text to copy';
      }
      setTimeout(() => { button.textContent = 'Copy'; }, 1800);
    });
    wrapper.append(button);
  });
}

// Content and nested navigation remain usable without JavaScript.
const printButton = document.querySelector('.print-button');
printButton.hidden = false;
printButton.addEventListener('click', () => window.print());

const contents = document.querySelector('.contents');
const mobile = window.matchMedia('(max-width: 860px)');
const setContents = () => { contents.open = !mobile.matches; };
setContents();
mobile.addEventListener('change', setContents);
contents.addEventListener('click', event => {
  if (mobile.matches && event.target.closest('a[href^="#"]')) contents.open = false;
});

const links = [...document.querySelectorAll('.contents a')];
const headings = [...document.querySelectorAll('.prose h2, .prose h3')];
const chapters = [...document.querySelectorAll('.chapter')];
let currentChapter = null;
let scheduled = false;
function markSection() {
  const heading = headings.findLast(node => node.getBoundingClientRect().top <= 120);
  const link = heading && links.find(node => decodeURIComponent(node.hash.slice(1)) === heading.id);
  const chapter = link?.closest('.chapter');
  if (chapter !== currentChapter) {
    if (currentChapter) {
      currentChapter.classList.remove('is-active');
      if (!currentChapter.contains(document.activeElement)) currentChapter.open = false;
    }
    if (chapter) {
      chapter.classList.add('is-active');
      chapter.open = true;
    }
    currentChapter = chapter;
  }
  for (const item of links) {
    if (item === link) item.setAttribute('aria-current', 'location');
    else item.removeAttribute('aria-current');
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

if (navigator.clipboard && window.isSecureContext) {
  document.querySelectorAll('.prose pre').forEach(pre => {
    const wrapper = document.createElement('div');
    wrapper.className = 'code-block';
    pre.before(wrapper);
    wrapper.append(pre);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-code';
    button.textContent = '复制';
    button.setAttribute('aria-label', '复制代码');
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(pre.querySelector('code').textContent);
        button.textContent = '已复制';
      } catch {
        button.textContent = '请选中文本复制';
      }
      setTimeout(() => { button.textContent = '复制'; }, 1800);
    });
    wrapper.append(button);
  });
}

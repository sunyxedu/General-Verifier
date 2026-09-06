// All pages, section links, and diagrams also work without JavaScript.
const printButton = document.querySelector('.print-button');
printButton.hidden = false;
printButton.addEventListener('click', () => window.print());

const siteMenu = document.querySelector('.site-menu');
const pageMenu = document.querySelector('.page-menu');
const mobile = window.matchMedia('(max-width: 760px)');
const setSiteMenu = () => { siteMenu.open = !mobile.matches; };
setSiteMenu();
mobile.addEventListener('change', setSiteMenu);

const links = [...pageMenu.querySelectorAll('a')];
const headings = [...document.querySelectorAll('.prose h2, .prose h3')];
let scheduled = false;
function markSection() {
  const heading = headings.findLast(node => node.getBoundingClientRect().top <= 120);
  for (const link of links) {
    if (heading && decodeURIComponent(link.hash.slice(1)) === heading.id) {
      link.setAttribute('aria-current', 'location');
    } else {
      link.removeAttribute('aria-current');
    }
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

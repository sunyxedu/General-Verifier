// The article, links and collapsible contents also work without JavaScript.
const printButton = document.querySelector('.print-button');
printButton.hidden = false;
printButton.addEventListener('click', () => window.print());

const contents = document.querySelector('.contents');
const mobile = window.matchMedia('(max-width: 960px)');
const setContents = () => { contents.open = !mobile.matches; };
setContents();
mobile.addEventListener('change', setContents);

const links = [...document.querySelectorAll('.contents a')];
const headings = [...document.querySelectorAll('.prose h2')];
let scheduled = false;
function markSection() {
  const heading = headings.findLast(node => node.getBoundingClientRect().top <= 160);
  for (const link of links) {
    const active = heading && decodeURIComponent(link.hash.slice(1)) === heading.id;
    if (active) link.setAttribute('aria-current', 'location');
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
markSection();

#!/usr/bin/env python3
"""Build the report into a small, self-contained GitHub Pages directory."""

from __future__ import annotations

import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import re
import subprocess
from string import Template
from urllib.parse import quote, unquote, urlsplit
from xml.etree import ElementTree as ET

from markdown_it import MarkdownIt
from markdown_it.token import Token


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "report"
DIAGRAMS = ("pipeline.svg", "contract.svg", "audit.svg", "reliance.svg")
ASSETS = (*DIAGRAMS, "report.css", "report.js")
# These H2 sections start topic pages. Earlier sections form the overview.
PAGE_SECTIONS = (
    ("getting-started.html", "Getting started", "Using GV"),
    ("workflow.html", "Verification workflow", "Using GV"),
    ("contracts.html", "Task contracts", "Design"),
    ("evidence.html", "Evidence and audit", "Design"),
    ("final-checks.html", "Independent final checks", "Design"),
    ("implementation.html", "Implementation", "Design"),
    ("results.html", "Reading the results", "Using GV"),
    ("limitations.html", "Scope and limitations", "Reference"),
    ("evaluation.html", "Evaluation", "Reference"),
    ("references.html", "Code and references", "Reference"),
)
HTML_PAGES = {"index.html", "report.html", *(page[0] for page in PAGE_SECTIONS)}
PUBLISHED = HTML_PAGES | {"report.md", ".nojekyll"} | {
    f"assets/{name}" for name in ASSETS
}


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True,
    ).strip()


def source_url(url: str, repository: str, revision: str,
               source_paths: set[str]) -> str:
    """Pin source references; accept only explicitly published local assets."""
    parsed = urlsplit(url)
    if parsed.scheme:
        if parsed.scheme not in {"https", "http", "mailto"}:
            raise ValueError(f"Unsupported link scheme: {url}")
        return url
    if parsed.netloc:
        raise ValueError(f"Use an explicit https URL: {url}")
    if not parsed.path:
        return url
    path = unquote(parsed.path)
    if path.startswith("../../"):
        relative = (REPORT / path).resolve().relative_to(ROOT).as_posix()
        if relative not in source_paths:
            raise ValueError(f"Source missing at reviewed commit: {relative}")
        target = f"https://github.com/{repository}/blob/{revision}/{quote(relative)}"
        return target + (f"#{parsed.fragment}" if parsed.fragment else "")
    if path not in PUBLISHED or (path not in HTML_PAGES and not (REPORT / path).is_file()):
        raise ValueError(f"Link is outside the published report: {url}")
    return url


def heading_title(inline: Token) -> str:
    return "".join(child.content for child in inline.children or []
                   if child.type in {"text", "code_inline"})


def parse_report(markdown: str) -> tuple[MarkdownIt, list[Token]]:
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = md.parse(markdown)
    if not tokens or tokens[0].type != "heading_open" or tokens[0].tag != "h1":
        raise ValueError("The report must begin with one H1 title.")
    ids: set[str] = set()
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        if token.tag == "h1" and index:
            raise ValueError("Only one H1 title is allowed.")
        title = heading_title(tokens[index + 1])
        slug = re.sub(r"[^\w\-\s]", "", title.lower())
        slug = re.sub(r"[\s_]+", "-", slug).strip("-") or "section"
        anchor = slug
        suffix = 2
        while anchor in ids:
            anchor = f"{slug}-{suffix}"
            suffix += 1
        ids.add(anchor)
        token.attrSet("id", anchor)
    return md, tokens


def render_tokens(md: MarkdownIt, original: list[Token], repository: str,
                  revision: str, source_paths: set[str],
                  destinations: dict[str, str], current_page: str,
                  *, full_document: bool = False,
                  promote_headings: bool = False) -> tuple[str, str]:
    # Each page, including the complete reading/print view, uses the same tokens.
    tokens = deepcopy(original)

    def link_url(url: str) -> str:
        url = source_url(url, repository, revision, source_paths)
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc or not parsed.fragment:
            return url
        if parsed.path and parsed.path not in HTML_PAGES:
            return url
        anchor = unquote(parsed.fragment)
        if anchor not in destinations:
            raise ValueError(f"Unknown report section: {url}")
        destination = destinations[anchor]
        if parsed.path and urlsplit(destination).path and parsed.path != urlsplit(destination).path:
            raise ValueError(f"Section is on another page: {url}")
        if full_document or urlsplit(destination).path == current_page:
            return f"#{anchor}"
        return destination

    chapters = []
    for index, token in enumerate(tokens):
        if promote_headings and token.type in {"heading_open", "heading_close"}:
            token.tag = f"h{int(token.tag[1:]) - 1}"
        if token.type == "heading_open":
            inline = tokens[index + 1]
            title, anchor = heading_title(inline), token.attrGet("id")
            if token.tag == "h2":
                chapters.append({"title": title, "anchor": anchor, "children": []})
            elif token.tag == "h3" and chapters:
                chapters[-1]["children"].append((title, anchor))
            link = Token("link_open", "a", 1)
            link.attrs = {"href": f"#{anchor}", "class": "heading-anchor",
                          "aria-label": f"Link to {title}"}
            symbol = Token("text", "", 0)
            symbol.content = "#"
            inline.children.extend([link, symbol, Token("link_close", "a", -1)])
        for child in token.children or []:
            for attribute in ("href", "src"):
                url = child.attrGet(attribute)
                if url:
                    child.attrSet(attribute, link_url(url))

    md.renderer.rules["table_open"] = lambda *_: (
        '<div class="table-wrap" role="region" aria-label="Table; scroll horizontally for more columns" '
        'tabindex="0"><table>\n'
    )
    md.renderer.rules["table_close"] = lambda *_: "</table></div>\n"
    # A fresh renderer prevents one page's image rule wrapping another's.
    render_image = MarkdownIt().renderer.rules["image"]

    def linked_image(tokens, index, options, env):
        source = tokens[index].attrGet("src") or ""
        src = escape(source, quote=True)
        if source in {f"assets/{name}" for name in DIAGRAMS}:
            ET.register_namespace("", "http://www.w3.org/2000/svg")
            diagram = ET.fromstring((REPORT / source).read_text(encoding="utf-8"))
            for link in diagram.iter("{http://www.w3.org/2000/svg}a"):
                href = link.get("href", "")
                parsed = urlsplit(href.removeprefix("../"))
                if not href.startswith("../") or parsed.path not in HTML_PAGES or not parsed.fragment:
                    raise ValueError(f"Diagram link must target a report section: {href}")
                link.set("href", link_url(href[3:]))
            label = escape(tokens[index].content, quote=True)
            svg = ET.tostring(diagram, encoding="unicode")
            return (
                '<span class="diagram-frame">'
                f'<span class="diagram-scroll" role="region" aria-label="{label}" '
                f'tabindex="0">{svg}</span>'
                '<span class="diagram-tools">'
                '<span class="diagram-hint">Scroll horizontally to follow the flow.</span>'
                f'<a href="{src}">Open full-size diagram ↗</a>'
                '</span></span>'
            )
        return (f'<a class="figure-link" href="{src}" title="Open full-size diagram">'
                f'{render_image(tokens, index, options, env)}</a>')

    md.renderer.rules["image"] = linked_image
    toc = []
    for chapter in chapters:
        title, anchor = escape(chapter["title"]), escape(chapter["anchor"])
        children = "".join(
            f'<li><a href="#{escape(child_anchor)}">{escape(child_title)}</a></li>'
            for child_title, child_anchor in chapter["children"]
        )
        nested = f'<ul>{children}</ul>' if children else ""
        toc.append(f'<li><a href="#{anchor}">{title}</a>{nested}</li>')
    return md.renderer.render(tokens, md.options, {}), "\n".join(toc)


def render_report(markdown: str, repository: str, revision: str,
                  source_paths: set[str]) -> tuple[str, str]:
    """Render a standalone document; also used by parser regression checks."""
    md, tokens = parse_report(markdown)
    destinations = {token.attrGet("id"): f'#{token.attrGet("id")}'
                    for token in tokens if token.type == "heading_open"}
    return render_tokens(md, tokens[3:], repository, revision, source_paths,
                         destinations, "", full_document=True)


def site_navigation(current_page: str) -> str:
    def item(path: str, title: str) -> str:
        current = ' aria-current="page"' if path == current_page else ""
        return f'<li><a href="{path}"{current}>{escape(title)}</a></li>'

    links = ['<ul>' + item("index.html", "Overview") + '</ul>']
    for group in ("Using GV", "Design", "Reference"):
        items = "".join(item(path, title) for path, title, section in PAGE_SECTIONS
                        if section == group)
        links.append(f'<p class="nav-group">{group}</p><ul>{items}</ul>')
    return "\n".join(links)


def render_site(markdown: str, repository: str, revision: str,
                source_paths: set[str]) -> dict[str, bytes]:
    md, tokens = parse_report(markdown)
    starts = [("index.html", heading_title(tokens[1]), 0)]
    expected = {title: path for path, title, _ in PAGE_SECTIONS}
    destinations = {}
    current = "index.html"
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        title = heading_title(tokens[index + 1])
        if token.tag == "h2" and title in expected:
            current = expected[title]
            starts.append((current, title, index))
        destinations[token.attrGet("id")] = f'{current}#{token.attrGet("id")}'
    if [page[0] for page in starts[1:]] != [page[0] for page in PAGE_SECTIONS]:
        raise ValueError("The report must contain each topic section once, in reading order.")

    template = Template((REPORT / "template.html").read_text(encoding="utf-8"))
    repo_url = f"https://github.com/{repository}"
    source = f"{repo_url}/blob/{git('rev-parse', 'HEAD')}/docs/report/report.md"
    pages = {}
    for index, (path, title, start) in enumerate(starts + [("report.html", "General Verifier", 0)]):
        complete = path == "report.html"
        end = starts[index + 1][2] if index + 1 < len(starts) and not complete else len(tokens)
        body, toc = render_tokens(
            md, tokens[start + 3:end], repository, revision, source_paths,
            destinations, path, full_document=complete,
            promote_headings=path not in {"index.html", "report.html"},
        )
        previous = starts[index - 1] if 0 < index < len(starts) else None
        following = starts[index + 1] if index + 1 < len(starts) else None
        pager = ""
        for relation, target in (("prev", previous), ("next", following)):
            if target:
                label = "Previous" if relation == "prev" else "Next"
                pager += (f'<a rel="{relation}" href="{target[0]}">'
                          f'<span>{label}</span>{escape(target[1])}</a>')
        pages[path] = template.substitute(
            title=escape(title), title_id=escape(tokens[start].attrGet("id")),
            tab_title=escape("General Verifier" if path in {"index.html", "report.html"}
                             else f"{title} | General Verifier"),
            body=body, toc=toc, navigation=site_navigation(path), pager=pager,
            repository=escape(repo_url), report_source=escape(source),
        ).encode()
    return pages


def build(output: Path, repository: str = "sunyxedu/General-Verifier") -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must have the form owner/name.")
    validation = json.loads((REPORT / "validation.json").read_text(encoding="utf-8"))
    revision = validation["reviewed_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("The reviewed implementation needs a full Git commit SHA.")
    source_paths = set(git("ls-tree", "-r", "--name-only", revision).splitlines())
    markdown = (REPORT / "report.md").read_text(encoding="utf-8")
    # Construct a publication allowlist in memory. No repository-wide copy,
    # recursive deletion, or dependency on API keys / the verifier package.
    payload = render_site(markdown, repository, revision, source_paths)
    payload.update({"report.md": markdown.encode(), ".nojekyll": b""})
    payload.update({f"assets/{name}": (REPORT / "assets" / name).read_bytes()
                    for name in ASSETS})
    output = output.absolute()
    if output.is_symlink() or ROOT.is_relative_to(output.resolve()):
        raise ValueError("Choose a separate output directory, not a source ancestor.")
    if output.exists():
        for existing in output.rglob("*"):
            relative = existing.relative_to(output).as_posix()
            if existing.is_symlink() or (existing.is_file() and relative not in PUBLISHED):
                raise ValueError(f"Use an empty output directory; unexpected file: {relative}")
    # Even a directory that only contains allowed filenames may be source.
    if output.resolve().is_relative_to(REPORT):
        raise ValueError("Output must not overwrite the report source directory.")
    for name, content in payload.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return output / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    parser.add_argument("--repository", default="sunyxedu/General-Verifier")
    args = parser.parse_args()
    try:
        destination = build(args.output, args.repository)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Report build failed: {exc}\n")
    print(f"Built {destination}")


if __name__ == "__main__":
    main()

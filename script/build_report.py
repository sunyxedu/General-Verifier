#!/usr/bin/env python3
"""Build the report into a small, self-contained GitHub Pages directory."""

from __future__ import annotations

import argparse
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
PUBLISHED = {"index.html", "report.md", ".nojekyll"} | {
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
    if path not in PUBLISHED or not (REPORT / path).is_file():
        raise ValueError(f"Link is outside the published report: {url}")
    return url


def render_report(markdown: str, repository: str, revision: str,
                  source_paths: set[str]) -> tuple[str, str]:
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = md.parse(markdown)
    if not tokens or tokens[0].type != "heading_open" or tokens[0].tag != "h1":
        raise ValueError("The report must begin with one H1 title.")
    tokens = tokens[3:]  # The page template supplies the report title.
    ids: set[str] = set()
    chapters = []
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            if token.tag == "h1":
                raise ValueError("Only one H1 title is allowed.")
            inline = tokens[index + 1]
            title = "".join(child.content for child in inline.children or []
                            if child.type in {"text", "code_inline"})
            slug = re.sub(r"[^\w\-\s]", "", title.lower())
            slug = re.sub(r"[\s_]+", "-", slug).strip("-") or "section"
            anchor = slug
            suffix = 2
            while anchor in ids:
                anchor = f"{slug}-{suffix}"
                suffix += 1
            ids.add(anchor)
            token.attrSet("id", anchor)
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
                    child.attrSet(attribute, source_url(
                        url, repository, revision, source_paths,
                    ))

    for token in tokens:
        for child in token.children or []:
            url = child.attrGet("href") or ""
            if url.startswith("#") and unquote(url[1:]) not in ids:
                raise ValueError(f"Unknown report section: {url}")

    # Tables remain keyboard-scrollable on narrow screens.
    md.renderer.rules["table_open"] = lambda *_: (
        '<div class="table-wrap" role="region" aria-label="Table; scroll horizontally for more columns" '
        'tabindex="0"><table>\n'
    )
    md.renderer.rules["table_close"] = lambda *_: "</table></div>\n"
    render_image = md.renderer.rules["image"]

    def linked_image(tokens, index, options, env):
        source = tokens[index].attrGet("src") or ""
        src = escape(source, quote=True)
        if source in {f"assets/{name}" for name in DIAGRAMS}:
            # Inline only the bundled diagrams so their step links work in
            # place. Each SVG remains a standalone, downloadable source file.
            ET.register_namespace("", "http://www.w3.org/2000/svg")
            diagram = ET.fromstring((REPORT / source).read_text(encoding="utf-8"))
            for link in diagram.iter("{http://www.w3.org/2000/svg}a"):
                href = link.get("href", "")
                prefix = "../index.html#"
                if not href.startswith(prefix):
                    raise ValueError(f"Diagram link must target a report section: {href}")
                anchor = unquote(href[len(prefix):])
                if anchor not in ids:
                    raise ValueError(f"Unknown diagram section: {href}")
                link.set("href", f"#{anchor}")
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
    for index, chapter in enumerate(chapters):
        title, anchor = escape(chapter["title"]), escape(chapter["anchor"])
        children = "".join(
            f'<li><a href="#{escape(child_anchor)}">{escape(child_title)}</a></li>'
            for child_title, child_anchor in chapter["children"]
        )
        if children:
            opened = " open" if index == 0 else ""
            toc.append(
                f'<li><details class="chapter" data-section="{anchor}"{opened}>'
                f'<summary><a class="chapter-link" href="#{anchor}">{title}</a></summary>'
                f'<ol class="sections">{children}</ol></details></li>'
            )
        else:
            toc.append(f'<li><a class="chapter-link" href="#{anchor}">{title}</a></li>')
    return md.renderer.render(tokens, md.options, {}), "\n".join(toc)


def build(output: Path, repository: str = "sunyxedu/General-Verifier") -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must have the form owner/name.")
    validation = json.loads((REPORT / "validation.json").read_text(encoding="utf-8"))
    revision = validation["reviewed_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("The reviewed implementation needs a full Git commit SHA.")
    source_paths = set(git("ls-tree", "-r", "--name-only", revision).splitlines())
    markdown = (REPORT / "report.md").read_text(encoding="utf-8")
    body, toc = render_report(markdown, repository, revision, source_paths)
    reviewed_date = validation["started_at_utc"][:10]
    repo_url = f"https://github.com/{repository}"
    template = Template((REPORT / "template.html").read_text(encoding="utf-8"))
    page = template.substitute(
        body=body, toc=toc, repository=escape(repo_url),
        revision=revision, short_revision=revision[:7],
        reviewed_date=escape(reviewed_date),
        report_source=f"{repo_url}/blob/{git('rev-parse', 'HEAD')}/docs/report/report.md",
    )

    # Construct a publication allowlist in memory. No repository-wide copy,
    # recursive deletion, or dependency on API keys / the verifier package.
    payload = {
        "index.html": page.encode(), "report.md": markdown.encode(),
        ".nojekyll": b"",
    }
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

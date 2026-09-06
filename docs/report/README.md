# General Verifier documentation site

[Read the documentation](https://hemlix.me/General-Verifier/) · [Markdown source](report.md)

The site opens with an overview: what GV does, a five-stage diagram whose actions link to their explanations, the two rules that shape the design, and a routing table for different readers. Fourteen topic pages follow in reading order, grouped in the site map as Start, Design, Guides, Internals, and Reference. All pages are in English.

The layout is the Hugo Book theme distributed with the AFL++ website. The left column holds search and the site map; the right column holds the current page's sections. At the theme's mobile breakpoint these become a navigation drawer and a section list opened from the header. Diagrams fit the content width and link to their full-size SVG. Text, links, fonts, and navigation work offline without JavaScript; search and code copying are optional JavaScript features.

## Publish with GitHub Pages

For the initial setup:

1. Open **Settings → Pages → Build and deployment → Source** in the repository and select **GitHub Actions**.
2. Push to `main`. If the commits are already pushed, use **Actions → Documentation → Run workflow**.
3. Wait for `build` and `deploy` to succeed, then open the link in the run summary. The URL also appears in the `github-pages` deployment environment.

This repository publishes to **<https://hemlix.me/General-Verifier/>**; the default GitHub Pages URL redirects there. A fork uses the URL in its own deployment summary.

Later changes to `docs/report/**`, the builder, its test, or the workflow publish automatically when pushed to `main`. Pull requests check the build and upload a downloadable site artifact; deployment happens after merge. Deployments use GitHub's automatically issued token and need no model API key or additional secret.

## Read locally

From the root of a full Git checkout:

```bash
uv run --no-project --with-requirements docs/report/requirements.txt python script/build_report.py
python3 -m http.server 8000 --bind 127.0.0.1 --directory _site
```

Open **<http://localhost:8000/>**, or open `_site/index.html` directly. The complete document is also available at `_site/report.html` for printing; it has no entry in the site map. The build installs two pinned Markdown parser dependencies, does not start the verifier, and does not call a model. The default output directory, `_site/`, is ignored by Git.

## How the site is organized

| Page | Group | Answers |
|---|---|---|
| Overview | | What is GV, how does a run flow through it, where should I start? |
| Getting started | Start | How do I install it, run the example, and read the result? |
| Architecture | Design | What are the components, roles, files, and entry points? |
| Claims and the claim graph | Design | What is a claim, how do tool results enter the graph, what do statuses mean? |
| Task contracts | Design | How does a claim become obligations, and how are they reviewed? |
| Evidence | Design | How is one obligation settled: methods, execution, gates, audit? |
| Verdicts and certificates | Design | How do outcomes become a status, a verdict, a certificate, a rank? |
| Roles and the eight axioms | Design | How are model roles constrained, and what are they told about evidence? |
| Capturing agent runs | Guides | How do I record and score my own runs? |
| Interactive verification | Guides | How does an agent get verification feedback while it works? |
| Identity, reuse, and revision | Internals | What makes two checks the same, and what does a revision invalidate? |
| Execution, resumption, and backends | Internals | Concurrency, resumption stamps, artifacts, and the two model backends |
| Reference | Reference | Commands, variables, files, formats, fields, code map |
| Scope and limitations | Reference | What a certificate does and does not establish; how to evaluate GV |

The sum example from `examples/sum/` is the running example on every page that needs one.

## Writing guidelines

Write for technical readers who are new to GV.

- Open each page with one or two sentences saying what problem the page's subject solves, then explain the mechanism. A reader arriving from a link should not need the previous page.
- Explain why a component exists before how it works. Choose implementation details that affect correctness, reproducibility, or cost, and say which problem each mechanism solves.
- Prefer short sentences, tables for parallel facts, and one running example. Keep code snippets focused; link to the implementation at the pinned revision with `../../path#Lnn` links, which the builder turns into GitHub links.
- Every claim about GV's behavior must come from the code. Distinguish measured results, explanatory examples, and policy choices.
- Keep the reference theme's typography, spacing, and navigation. Do not add a cover, subtitles, metadata strips, or a single-page reading link.

The structural models are the [AFL++ documentation](https://aflplus.plus/docs/) (an overview diagram whose steps link to detail pages, and a site map separate from the page's sections), the [Rust Book](https://doc.rust-lang.org/stable/book/ch04-01-what-is-ownership.html) (one running example carried through a difficult concept), and [Tradingflow](https://bridgekat.github.io/tradingflow/tradingflow/index.html) (a concise index that routes readers).

## Diagrams

Diagrams are native SVG in `assets/`, generated from short scripts and edited by hand as needed. Text stays text so it remains sharp and searchable.

| Figure | Page | Reader's question |
|---|---|---|
| `pipeline.svg` | Overview | What happens to a run, and where do I read about each stage? |
| `components.svg` | Architecture | Which components exist and how does data move between them? |
| `graph.svg` | Claims | What does a claim graph look like for the sum example? |
| `contract.svg` | Task contracts | When is a contract accepted, repaired, or abandoned? |
| `audit.svg` | Evidence | When is a result accepted, redone, or rejected? |
| `ladder.svg` | Evidence | In which order do strength tiers run, and what unlocks a weaker tier? |
| `settle.svg` | Verdicts | How do obligation outcomes become a claim status? |

Conventions: rectangles are actions or objects, diamonds are decisions, arrow labels state the branch condition, dashed borders mark optional work, and shaded boxes mark mechanical steps or terminal states. Overview actions link to `../page.html#anchor`; the builder validates every target and rewrites the links for each page. Prefix every SVG id with the diagram name so ids never collide when several diagrams share a page. After changing a figure, check text bounds and arrow crossings at reading size and confirm that every depicted branch matches the implementation.

## Source files and the pinned implementation

| File | Purpose |
|---|---|
| [report.md](report.md) | The single Markdown source for the overview, the topic pages, and the complete view |
| `validation.json` | Maintainer record of the reviewed implementation commit; read during the build, excluded from publication |
| [assets/](assets/) | Diagrams, the bundled theme and fonts, GV overrides, interactions, and license notices |
| [template.html](template.html) | Shared page structure, site map, and current-page sections |
| [build_report.py](../../script/build_report.py) | Splits topics, resolves section links, renders navigation, pins code links |
| [report.yml](../../.github/workflows/report.yml) | Builds, checks, deploys, and prints the published URL |

Code links are pinned to `reviewed_commit` in `validation.json`, and the builder checks that each linked file exists at that revision. Text and layout changes may keep the existing pin. If a page starts describing new behavior, review the code and run its tests before updating the pin and the prose together; never update only the commit.

## Page boundaries and links

`PAGE_SECTIONS` in the builder names the H2 headings that start topic pages and the group each page belongs to; content before the first topic forms the overview. Use H3 headings for sections inside a topic; the builder shows them as H2 on that page. Keep the H2 headings in `report.md` in the same order as `PAGE_SECTIONS`.

Write ordinary `#anchor` links in Markdown. All headings get their ids before splitting, so the builder can route a link to the owning page while the complete view keeps it in-document. Adding a topic means adding a row to `PAGE_SECTIONS`; the publication allowlist and the site map derive from it.

## Theme and navigation

`assets/book.css` comes from the [AFL++ website source](https://github.com/AFLplusplus/Website/tree/1cb95db7f3667cc42d5ae2863931c15d7e81d0b7), which uses Hugo Book. Roboto and Roboto Mono are bundled locally; the exact sources and licenses are in [assets/licenses/NOTICE.txt](assets/licenses/NOTICE.txt). GV's overrides live in `assets/report.css` and support 320px screens, accessible native menu controls, responsive inline SVGs, and printing. Stylesheet, script, and full-size diagram URLs carry content fingerprints so a layout change never combines new HTML with cached old assets.

Search uses an index built from the same headings and text. Each result opens its owning page and section. The index is embedded as inert JSON, and results are created with `textContent`, so code examples cannot become markup.

## Check the build

```bash
uv run --no-project --with-requirements docs/report/requirements.txt python -m unittest discover -s tests -p test_report_build.py -v
```

The checks cover every link and anchor across all HTML pages, embedded and standalone SVGs, and search results, plus the complete view, pinned code links, duplicate anchors, and publication boundaries. Renaming a section that a diagram targets requires updating the SVG link; an unresolved destination stops the build. The builder writes only explicitly listed files and refuses to overwrite a directory containing anything else; use `--output /path/to/empty-directory` for a fresh location.

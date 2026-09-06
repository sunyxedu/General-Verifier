# General Verifier technical report

[Read the report](https://hemlix.me/General-Verifier/) · [Markdown source](report.md)

The website opens with a short explanation of GV and a clickable workflow. Focused pages cover getting started, the workflow, task contracts, evidence audit, independent final checks, implementation, results, limitations, evaluation, and references. All pages are in English.

The left navigation is a site map; the right navigation lists sections on the current page. Both collapse on narrow screens. The site also provides code copying, section links, and print styles. Wide diagrams scroll within their own viewport and have a full-size link. Text, diagrams, and links work offline without JavaScript; native menus remain expandable.

## Publish with GitHub Pages

For the initial setup:

1. Open **Settings → Pages → Build and deployment → Source** in the repository and select **GitHub Actions**.
2. Push the report commits to `main`. If they are already pushed, use **Actions → Technical report → Run workflow**.
3. Wait for `build` and `deploy` to succeed, then open **Read the report** in the run summary. The URL also appears in the `github-pages` deployment environment.

This repository's published URL is:

**<https://hemlix.me/General-Verifier/>**

The default GitHub Pages URL, <https://sunyxedu.github.io/General-Verifier/>, redirects there. For a fork or a different domain configuration, use the actual URL in the deployment summary.

Subsequent changes to the report, diagrams, or site build files publish automatically when pushed to `main`. Pull requests check the build and upload a downloadable site artifact; deployment happens after merge. The deployment summary uses the URL returned by GitHub.

Enabling Pages for the first time requires repository settings access and a repository and account that support Pages. The normal workflow `GITHUB_TOKEN` cannot enable it by itself. Later deployments use GitHub's automatically issued token and need no model API key, manually created personal access token, or additional deployment secret. See the [GitHub custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) and [configure-pages permissions for initial enablement](https://github.com/actions/configure-pages/blob/v5/action.yml).

## Read locally

From the root of a full Git checkout:

```bash
uv run --no-project --with-requirements docs/report/requirements.txt python script/build_report.py
python3 -m http.server 8000 --bind 127.0.0.1 --directory _site
```

Open **<http://localhost:8000/>** or open `_site/index.html` directly in a browser. Use **Print this page** at the bottom, or the browser's print menu, to export the current topic. Choose **Read as one page** in the site map to read or print the complete report. Content, navigation, and images require neither JavaScript nor an external CDN.

The site build installs only two pinned Markdown parser dependencies. It does not start the verifier or invoke models. The default output directory, `_site/`, is ignored by Git.

## Writing and organization

Write for technical readers who are new to GV:

- Use English throughout the report, diagrams, navigation, and interface labels.
- Open with the project's purpose, what it reads and produces, and a workflow that links to explanations. Put the runnable example on Getting started; preserve every task requirement, including output format.
- Give each topic a short introduction that works when readers arrive directly from a link. Introduce the shared example where needed instead of assuming that every reader has read the preceding page.
- Explain why a component exists before describing how it works. Choose implementation details that affect correctness, reproducibility, or cost, and explain the problem each mechanism solves.
- Reuse the same example when developing a concept. Keep code snippets focused and link to the complete implementation at a fixed revision.
- Distinguish measured results, explanatory examples, and proposed experiments. When discussing a semantic limitation, say which conclusion it limits.
- Keep the layout plain: a white background, normal-weight headings, site navigation, and a separate page contents list. Avoid numbered chapter headings, a report-style cover, subtitles, and metadata strips.
- The overview's browser title is General Verifier. Topic tabs use the topic name followed by General Verifier. Keep commit IDs, dates, regression counts, and validation records out of the published prose; retain the implementation pin internally for source links.

The main layout reference is [AFL++ documentation](https://aflplus.plus/docs/): an overview introduces the workflow, actions link to detailed explanations, and site navigation is separate from the current page's contents. Its [in-depth guide](https://aflplus.plus/docs/fuzzing_in_depth/) connects operational steps to their reasons. The [Rust Book](https://doc.rust-lang.org/stable/book/ch04-01-what-is-ownership.html) and [Tradingflow](https://bridgekat.github.io/tradingflow/tradingflow/index.html) also show how to build explanations around concrete examples. Claims about GV itself must come from its implementation.

## Diagram conventions

The diagrams answer different questions:

| Figure | Reader's question |
|---|---|
| `pipeline.svg` | What happens to a run, and where can I read about each stage? |
| `contract.svg` | When are task checks accepted, repaired, or left unresolved? |
| `audit.svg` | When is evidence accepted, retried, or rejected? |
| `reliance.svg` | Why can an intermediate claim pass locally while its premise remains unresolved? |

Use action labels and named outputs. Group related actions under a stage heading. Keep the overview readable on its own, then expand its difficult decisions in separate figures near their explanations. In control-flow diagrams, diamonds ask questions, arrow labels state branch conditions, and return arrows show the exact operation repeated. Show budget exhaustion wherever a retry appears. The overview uses a dashed border for optional selection and explains that convention in its legend; color carries no essential meaning.

The [AFL++ overview](https://aflplus.plus/docs/) provided the model for grouped stages, required/optional notation, and links from steps to detailed documentation. Its [overview SVG source](https://github.com/AFLplusplus/AFLplusplus/blob/dev/docs/resources/0_fuzzing_process_overview.drawio.svg) is a diagrams.net export with embedded editor data. GV's figures are native SVG source files in `assets/`; edit their elements and labels directly. Keep text as SVG text so it stays sharp when enlarged or printed.

Overview actions link to `../topic.html#section-anchor` in the standalone SVG. The builder validates both the topic and the anchor. Embedded actions navigate to the same destination; in the complete report, they stay within the document. Prefix every SVG ID with its diagram name to prevent collisions across figures. The bundled SVGs are trusted repository assets and are embedded directly into the page.

After changing a figure, inspect it at normal reading size and in print. Check text bounds, arrow crossings, branch labels, keyboard access, and the narrow-screen scroll view. Confirm that each depicted branch agrees with the reviewed implementation. A diagram may summarize a mechanism, but its caption or nearby text must state the scope of that summary.

## Source files and implementation version

| File | Purpose |
|---|---|
| [report.md](report.md) | Single source for the overview, topic pages, and complete report; readable on GitHub |
| `validation.json` | Maintainer record of the reviewed implementation, test results, and fingerprints; read during the build and excluded from publication |
| [assets/](assets/) | Four SVG diagrams, site styles, and interactions |
| [template.html](template.html) | Shared page structure, site map, and current-page contents |
| [build_report.py](../../script/build_report.py) | Split topics, route section links, render navigation, and pin code references |
| [report.yml](../../.github/workflows/report.yml) | Build, check, deploy, and publish the report URL |

The historical regression record belongs in `validation.json`, which is excluded from the published files. Publishing the website does not rerun models, update research findings, or turn historical test results into validation of current code. The builder pins implementation links to that snapshot and checks that each referenced file exists at that Git revision.

Text and layout changes may retain the existing implementation snapshot. If the report begins describing new behavior, review the relevant code and run its tests before updating the report, revision, and measured results together. Do not update only the commit or test count.

## Page boundaries and links

`report.md` remains the only prose source. `PAGE_SECTIONS` in the builder names the H2 headings that start topic pages. Content before the first topic forms the overview. Use H3 headings for subsections inside a topic; the builder displays them as H2 on that page. Keep the topic headings in the configured reading order.

Use ordinary `#section-anchor` links in Markdown. The builder resolves them to the owning topic page, while the complete reading view keeps them in the same document. All headings receive their IDs before splitting, so links are identical across the Markdown, topic pages, and complete view. When adding a topic, update `PAGE_SECTIONS`; the publication allowlist and navigation derive from that list.

## Check the build

```bash
uv run --no-project --with-requirements docs/report/requirements.txt python -m unittest discover -s tests -p test_report_build.py -v
```

These checks cover links and target anchors across every HTML page and standalone SVG, the complete reading view, historical code links, duplicate anchors, and publication boundaries. Renaming a section targeted by a diagram requires updating the SVG link; an unresolved destination stops the build. The output directory contains only explicitly listed report files. The builder refuses to overwrite a directory containing unexpected files. Use `--output /path/to/empty-directory` to choose a fresh output location.

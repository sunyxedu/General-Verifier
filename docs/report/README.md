# General Verifier technical report

[Read the report](https://hemlix.me/General-Verifier/) · [Markdown source](report.md)

The report follows the repository's sum example through a complete verification run, then explains the architecture, three core design decisions, key implementation mechanisms, result interpretation, limitations, and evaluation. It is written in English and includes four diagrams and links to the reviewed implementation.

The published website provides nested navigation, clickable workflow steps, section links, code copying, mobile layouts, and print styles. Wide diagrams scroll within their own viewport and have a full-size link. The text, diagrams, and navigation also work offline without JavaScript.

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

Open **<http://localhost:8000/>** or open `_site/index.html` directly in a browser. Use **Print / PDF** at the top of the page, or the browser's print menu, to export a copy. Content, navigation, and images require neither JavaScript nor an external CDN.

The site build installs only two pinned Markdown parser dependencies. It does not start the verifier or invoke models. The default output directory, `_site/`, is ignored by Git.

## Writing and organization

Write for technical readers who are new to GV:

- Use English throughout the report, diagrams, navigation, and interface labels.
- Start with one complete example that connects inputs, processing, and outputs before introducing internal terminology. Preserve every requirement of the real task, including output format.
- Explain why a component exists before describing how it works. Choose implementation details that affect correctness, reproducibility, or cost, and explain the problem each mechanism solves.
- Reuse the same example when developing a concept. Keep code snippets focused and link to the complete implementation at a fixed revision.
- Distinguish measured results, explanatory examples, and proposed experiments. When discussing a semantic limitation, say which conclusion it limits.
- Keep the page focused on the text. Chapters should follow the learning sequence; nested navigation should help readers find details. Keep version metadata and tool links brief, and leave the detailed validation manifest out of the public site.

The organization draws on [Tradingflow's introduction](https://bridgekat.github.io/tradingflow/tradingflow/index.html) and [graph module](https://bridgekat.github.io/tradingflow/tradingflow_graph/index.html), which lead from examples into abstractions and implementation; the [Rust Book's ownership chapter](https://doc.rust-lang.org/stable/book/ch04-01-what-is-ownership.html), which develops rules through connected examples; and the [AFL++ documentation](https://aflplus.plus/docs/) and [AFL technical notes](https://aflplus.plus/docs/technical_details/), which connect workflow, mechanisms, and implementation choices. Claims about GV itself are grounded in the repository code and tests cited in the report.

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

Overview actions link to `../index.html#section-anchor` in the standalone SVG. The site builder validates those destinations and rewrites them to in-page links when embedding the diagram. Prefix every SVG ID with its diagram name to prevent collisions across figures. The bundled SVGs are trusted repository assets and are embedded directly into the page.

After changing a figure, inspect it at normal reading size and in print. Check text bounds, arrow crossings, branch labels, keyboard access, and the narrow-screen scroll view. Confirm that each depicted branch agrees with the reviewed implementation. A diagram may summarize a mechanism, but its caption or nearby text must state the scope of that summary.

## Source files and implementation version

| File | Purpose |
|---|---|
| [report.md](report.md) | Single source for the report body; readable directly on GitHub |
| `validation.json` | Maintainer record of the reviewed implementation, test results, and fingerprints; read during the build and excluded from publication |
| [assets/](assets/) | Four SVG diagrams, site styles, and interactions |
| [template.html](template.html) | Page structure and implementation version metadata |
| [build_report.py](../../script/build_report.py) | Render the page and navigation; validate and pin code references |
| [report.yml](../../.github/workflows/report.yml) | Build, check, deploy, and publish the report URL |

The report's **469 tests** refer to the implementation snapshot recorded in `validation.json`. Publishing the website does not rerun models, update research findings, or turn historical test results into validation of current code. The builder pins implementation links to that snapshot and checks that each referenced file exists at that Git revision.

Text and layout changes may retain the existing implementation snapshot. If the report begins describing new behavior, review the relevant code and run its tests before updating the report, revision, and measured results together. Do not update only the commit or test count.

## Check the build

```bash
uv run --no-project --with-requirements docs/report/requirements.txt python -m unittest discover -s tests -p test_report_build.py -v
```

These checks cover page and embedded diagram references, historical code links, duplicate anchors, and publication boundaries. Renaming a section targeted by a diagram requires updating the SVG link; an unresolved destination stops the build. The output directory contains only explicitly listed report files. The builder refuses to overwrite a directory containing unexpected files. Use `--output /path/to/empty-directory` to choose a fresh output location.

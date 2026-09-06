# General Verifier 技术报告

[阅读正文](report.md) · [网页版](https://hemlix.me/General-Verifier/) · [验证记录](validation.json)

这份报告从仓库的求和样例出发，依次说明一次完整验证、系统结构、三个关键设计、关键实现、结果的使用、通用性边界和验证依据。正文为中文，附英文摘要、两张图和固定版本的代码索引。

网页版已发布，提供分层目录、章节链接、代码复制、手机排版和打印样式。正文与目录也可以离线阅读，无需 JavaScript。

## 发布到 GitHub：首次设置一次，之后自动更新

1. 在仓库打开 **Settings → Pages → Build and deployment → Source**，选择 **GitHub Actions**。
2. 把这些提交推送到 `main`。如果已经推送过，在 **Actions → Technical report → Run workflow** 手动运行一次。
3. 等待 `build` 和 `deploy` 成功，打开运行摘要里的 **Read the report**。同一地址也会显示在 `github-pages` deployment environment 中。

本仓库当前使用的网页地址是：

**<https://hemlix.me/General-Verifier/>**

GitHub Pages 的默认入口 <https://sunyxedu.github.io/General-Verifier/> 会跳转到该地址。其他仓库或域名设置下，请以 deployment 摘要显示的实际地址为准。

以后修改报告、图表或网站构建文件并推送到 `main`，workflow 就会重新发布，网址保持不变。Pull request 只检查构建并上传可下载的站点产物；合入主分支后才部署。部署摘要使用 GitHub 返回的实际地址，因此 fork 或自定义域名应以摘要为准。

首次启用 Pages 需要仓库设置权限，且仓库及账号需要支持 Pages；普通 workflow 的 `GITHUB_TOKEN` 不能自行完成初次启用。后续部署使用 GitHub 自动签发的 token，无需模型 API key、手动创建 PAT 或配置额外发布密钥。参见 [GitHub 自定义 Pages workflow 文档](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) 和 [configure-pages 的初次启用权限说明](https://github.com/actions/configure-pages/blob/v5/action.yml)。

## 本地阅读

在完整的 Git checkout 根目录运行：

```bash
uv run --no-project --with-requirements docs/report/requirements.txt python script/build_report.py
python3 -m http.server 8000 --bind 127.0.0.1 --directory _site
```

打开 **<http://localhost:8000/>**。也可以直接用浏览器打开 `_site/index.html`。点击页首的 **打印 / PDF** 即可导出；浏览器菜单中的打印功能同样可用。正文、目录和图片不依赖 JavaScript 或外部 CDN。

网站构建只安装两个固定版本的 Markdown 解析依赖，不启动验证器、不调用模型。默认输出目录 `_site/` 已被 Git 忽略。

## 维护正文的写法

这份报告面向尚不熟悉 GV 的技术读者。更新时保持以下阅读顺序与约定：

- 从一个完整例子建立输入、处理过程和输出的关系，再引入内部术语。例子应遵守真实任务的全部要求，包括文件格式。
- 先讲组件为什么存在，再讲它如何工作。实现细节优先选择影响正确性、复现或成本的地方，并说明省略该机制会遇到什么问题。
- 同一概念尽量沿用同一例子。代码片段只保留正在解释的部分，旁边提供完整实现的固定版本链接。
- 把实测结果、解释性示例和未来实验明确区分。讨论语义判断的边界时，说明它具体限制了哪一种结论。
- 页面以正文为中心。一级目录对应理解顺序，二级目录帮助查找细节；版本、来源和工具入口保持简洁。

阅读组织参考了 [Tradingflow 的入口](https://bridgekat.github.io/tradingflow/tradingflow/index.html)及其 [graph 模块](https://bridgekat.github.io/tradingflow/tradingflow_graph/index.html)：先走完例子，再进入抽象与实现；[Rust Book 的 ownership 章节](https://doc.rust-lang.org/stable/book/ch04-01-what-is-ownership.html)：用连续例子解释规则出现的原因；以及 [AFL++ 文档入口](https://aflplus.plus/docs/)与 [AFL 技术说明](https://aflplus.plus/docs/technical_details/)：把流程、机制和具体实现相互连接。GV 的行为依据仍是正文引用的仓库代码与测试。

## 内容与实现版本

| 文件 | 用途 |
|---|---|
| [report.md](report.md) | 唯一正文源文件；GitHub 上也可以直接阅读 |
| [validation.json](validation.json) | 审查的实现 commit、测试结果与输入指纹 |
| [assets/](assets/) | 两张 SVG 图，以及网站样式和交互 |
| [template.html](template.html) | 网页结构与报告版本信息 |
| [build_report.py](../../script/build_report.py) | 生成网页、目录，并核对及固定代码引用 |
| [report.yml](../../.github/workflows/report.yml) | 构建、检查、部署与发布链接 |

报告中的 **469 项测试**对应 `validation.json` 记录的实现快照。网站发布不会自动重跑模型、更新报告的研究结论，或把旧测试结果说成当前代码的验证结果。构建器把正文中的仓库代码链接固定到该快照，并检查引用文件确实存在于那个 Git 版本。

只改文字或排版时，可以保留原实现快照。若报告开始描述新实现，应先审查相应代码并运行测试，再一起更新正文、`validation.json` 的版本与实测结果。不要仅替换 commit 或测试数量。

## 检查构建

```bash
uv run --no-project --with-requirements docs/report/requirements.txt python -m unittest discover -s tests -p test_report_build.py -v
```

这些检查验证页面引用、历史代码链接、重复标题锚点及发布目录边界。发布目录只包含明确列出的报告文件；构建器遇到额外文件会拒绝覆盖，避免把其他工作目录的内容一起发布。使用 `--output /path/to/empty-directory` 可以指定新的输出位置。

# General Verifier 技术报告

[阅读正文](report.md) · [网页版](https://sunyxedu.github.io/General-Verifier/) · [验证记录](validation.json)

这份报告从“如何判断代理真的完成了任务”出发，介绍 GV 的架构、检查条件、执行与证据审查、状态传播、复用与恢复机制。它也讨论 `clause`、`refutable_by_absence` 的通用性边界，并给出后续实验设计。正文为中文，附英文摘要和两张架构图。

网页版提供目录、固定版本的代码引用、手机排版和打印样式。上面的网页地址在首次 GitHub Pages 部署成功后生效；本地构建现在即可阅读。

## 发布到 GitHub：首次设置一次，之后自动更新

1. 在仓库打开 **Settings → Pages → Build and deployment → Source**，选择 **GitHub Actions**。
2. 把这些提交推送到 `main`。如果已经推送过，在 **Actions → Technical report → Run workflow** 手动运行一次。
3. 等待 `build` 和 `deploy` 成功，打开运行摘要里的 **Read the report**。同一地址也会显示在 `github-pages` deployment environment 中。

本仓库未配置自定义域名时，预期地址是：

**<https://sunyxedu.github.io/General-Verifier/>**

以后修改报告、图表或网站构建文件并推送到 `main`，workflow 就会重新发布，网址保持不变。Pull request 只检查构建并上传可下载的站点产物；合入主分支后才部署。部署摘要使用 GitHub 返回的实际地址，因此 fork 或自定义域名应以摘要为准。

首次启用 Pages 需要仓库设置权限，且仓库及账号需要支持 Pages；普通 workflow 的 `GITHUB_TOKEN` 不能自行完成初次启用。后续部署使用 GitHub 自动签发的 token，无需模型 API key、手动创建 PAT 或配置额外发布密钥。参见 [GitHub 自定义 Pages workflow 文档](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) 和 [configure-pages 的初次启用权限说明](https://github.com/actions/configure-pages/blob/v5/action.yml)。

## 本地阅读

在完整的 Git checkout 根目录运行：

```bash
uv run --no-project --with-requirements docs/report/requirements.txt python script/build_report.py
python3 -m http.server 8000 --bind 127.0.0.1 --directory _site
```

打开 **<http://localhost:8000/>**。也可以直接用浏览器打开 `_site/index.html`。点击页首的 **打印 / 存为 PDF** 即可导出；浏览器菜单中的打印功能同样可用。正文、目录和图片不依赖 JavaScript 或外部 CDN。

网站构建只安装两个固定版本的 Markdown 解析依赖，不启动验证器、不调用模型。默认输出目录 `_site/` 已被 Git 忽略。

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

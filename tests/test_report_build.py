"""Publication checks; also runnable with stdlib unittest and docs dependencies."""

from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

from script.build_report import HTML_PAGES, PUBLISHED, ROOT, build, render_report, source_url


class PageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.urls = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        for field in ("href", "src"):
            if field in attrs:
                self.urls.append(attrs[field])


class ReportBuildTests(unittest.TestCase):
    def test_full_site_has_only_report_files_and_resolved_local_links(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            build(output)
            first_build = {name: (output / name).read_bytes() for name in PUBLISHED}
            documents = {}
            for name in PUBLISHED:
                if name.endswith((".html", ".svg")):
                    parser = PageLinks()
                    parser.feed(first_build[name].decode())
                    self.assertEqual(len(parser.ids), len(set(parser.ids)), name)
                    documents[(output / name).resolve()] = parser
            for path, parser in documents.items():
                for url in parser.urls:
                    parsed = urlsplit(url)
                    if parsed.scheme or parsed.netloc:
                        continue
                    target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
                    self.assertTrue(target.is_relative_to(output), (path.name, url))
                    self.assertTrue(target.is_file(), (path.name, url))
                    if parsed.fragment:
                        self.assertIn(unquote(parsed.fragment), documents[target].ids, (path.name, url))
            self.assertEqual(
                {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()},
                PUBLISHED,
            )
            build(output)
            self.assertEqual({name: (output / name).read_bytes() for name in PUBLISHED}, first_build)

    def test_topic_navigation_and_single_page_reading_share_the_same_content(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            build(output)
            complete = PageLinks()
            complete.feed((output / "report.html").read_text())
            # Every topic section is available in the complete reading/print view.
            for name in HTML_PAGES - {"report.html"}:
                page = PageLinks()
                page.feed((output / name).read_text())
                self.assertTrue(set(page.ids).issubset(complete.ids), name)
            overview = PageLinks()
            overview.feed((output / "index.html").read_text())
            self.assertIn("contracts.html#task-contracts", overview.urls)
            self.assertIn("results.html#five-verdicts", overview.urls)
            self.assertIn("#task-contracts", complete.urls)
            self.assertIn("#five-verdicts", complete.urls)
            # In the complete view, content links stay within that document.
            self.assertNotIn("contracts.html#task-contracts", complete.urls)

    def test_implementation_links_are_pinned_to_reviewed_history(self):
        revision = "a" * 40
        path = "verifier/core/model.py"
        link = source_url(f"../../{path}#L10", "owner/project", revision, {path})
        self.assertEqual(link, f"https://github.com/owner/project/blob/{revision}/{path}#L10")
        with self.assertRaisesRegex(ValueError, "missing at reviewed commit"):
            source_url("../../verifier/new_module.py", "owner/project", revision, {path})

    def test_arbitrary_local_files_cannot_enter_publication_via_links(self):
        for path in ("requirements.txt", ".env", "/etc/passwd", "assets/../template.html"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                source_url(path, "owner/project", "a" * 40, set())

    def test_existing_output_with_extra_file_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            marker = output / "keep.txt"
            marker.write_text("unrelated content")
            with self.assertRaisesRegex(ValueError, "unexpected file"):
                build(output)
            self.assertEqual(marker.read_text(), "unrelated content")
            self.assertFalse((output / "index.html").exists())

    def test_output_cannot_overwrite_source_or_follow_asset_symlink(self):
        with self.assertRaisesRegex(ValueError, "source ancestor"):
            build(ROOT)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            target = Path(directory) / "elsewhere"
            output.mkdir()
            target.mkdir()
            (output / "assets").symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "unexpected file"):
                build(output)
            self.assertEqual(list(target.iterdir()), [])

    def test_duplicate_titles_get_unique_anchors_and_raw_html_is_not_executed(self):
        body, toc = render_report(
            '# Title\n\n## 示例\n\n<script>alert(1)</script>\n\n## 示例\n',
            "owner/project", "a" * 40, set(),
        )
        self.assertIn('id="示例"', body)
        self.assertIn('id="示例-2"', body)
        self.assertNotIn("<script>", body)
        self.assertIn('href="#示例-2"', toc)

    def test_broken_section_reference_stops_the_build(self):
        with self.assertRaisesRegex(ValueError, "Unknown report section"):
            render_report("# Title\n\n## Section\n\n[missing](#nope)",
                          "owner/project", "a" * 40, set())


if __name__ == "__main__":
    unittest.main()

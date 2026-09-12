#!/usr/bin/env python3
"""Render the package A2D Markdown report as a self-contained PDF."""
from __future__ import annotations

import base64
import mimetypes
import re
import subprocess
import tempfile
from pathlib import Path

import markdown


REPORT = Path("/home/bjtc/Sophix/lingbot-vla/output/package_a2d_lora_4090/offline_analysis/package_a2d_evaluation_report.md")
PDF = REPORT.with_suffix(".pdf")
EMBEDDED_HTML = REPORT.with_name("package_a2d_evaluation_report_embedded.html")


def image_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    source = REPORT.read_text(encoding="utf-8")
    html_body = markdown.markdown(source, extensions=["tables", "sane_lists"])

    def replace_image(match: re.Match[str]) -> str:
        alt, src = match.group(1), match.group(2)
        path = Path(src)
        if not path.is_absolute():
            path = REPORT.parent / path
        if not path.exists():
            raise FileNotFoundError(f"Referenced image does not exist: {path}")
        return f'<img src="{image_data_uri(path)}" alt="{alt}">' 

    html_body = re.sub(r'<img src="([^"]+)" alt="([^"]*)">', lambda m: f'<img src="{image_data_uri(Path(m.group(1)))}" alt="{m.group(2)}">' if Path(m.group(1)).is_absolute() else m.group(0), html_body)
    # markdown emits alt before src in some versions; normalize both forms.
    html_body = re.sub(r'<img alt="([^"]*)" src="([^"]+)">', replace_image, html_body)
    html_body = re.sub(r'<img src="([^"]+)" alt="([^"]*)">', lambda m: f'<img src="{image_data_uri(Path(m.group(1)))}" alt="{m.group(2)}">', html_body)

    full_html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 16mm 15mm 17mm; }}
body {{ font-family: "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; color: #202124; line-height: 1.48; font-size: 10.5pt; }}
h1 {{ font-size: 22pt; border-bottom: 2px solid #333; padding-bottom: 5pt; }}
h2 {{ font-size: 15pt; margin-top: 18pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; }}
h3 {{ font-size: 12pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 7pt 0 12pt; font-size: 8.4pt; page-break-inside: avoid; }}
th, td {{ border: 1px solid #999; padding: 3px 5px; vertical-align: top; }}
th {{ background: #eceff1; }}
img {{ display: block; max-width: 100%; max-height: 190mm; margin: 8pt auto 12pt; page-break-inside: avoid; }}
code {{ font-family: "DejaVu Sans Mono", monospace; font-size: 8.3pt; overflow-wrap: anywhere; }}
li {{ margin: 2pt 0; }}
p {{ margin: 5pt 0; }}
</style></head><body>{html_body}</body></html>'''
    EMBEDDED_HTML.write_text(full_html, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="package_pdf_") as profile:
        subprocess.run([
            "/usr/bin/google-chrome", "--headless", "--no-sandbox", "--disable-gpu",
            f"--user-data-dir={profile}", f"--print-to-pdf={PDF}", str(EMBEDDED_HTML)
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(PDF)


if __name__ == "__main__":
    main()

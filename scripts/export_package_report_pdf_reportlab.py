#!/usr/bin/env python3
"""Create an offline PDF report with PNGs embedded as PDF image objects."""
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path("/home/bjtc/Sophix/lingbot-vla/output/package_a2d_lora_4090/offline_analysis")
MD = ROOT / "package_a2d_evaluation_report.md"
PDF = ROOT / "package_a2d_evaluation_report_full.pdf"


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(text: str) -> str:
    text = esc(text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def table_from(lines: list[str], styles):
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append([Paragraph(inline(c), styles["TableCell"]) for c in cells])
    if not rows:
        return None
    widths = [None] * len(rows[0])
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf2")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#8a8a8a")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def main() -> None:
    # STSong-Light is a standard PDF CID font with broad Simplified Chinese
    # coverage and keeps the generated PDF portable across offline systems.
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    ss = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("Title", parent=ss["Title"], fontName="STSong-Light", fontSize=21, leading=27, alignment=TA_CENTER, spaceAfter=12),
        "H2": ParagraphStyle("H2", parent=ss["Heading2"], fontName="STSong-Light", fontSize=14, leading=19, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#202124")),
        "Body": ParagraphStyle("Body", parent=ss["BodyText"], fontName="STSong-Light", fontSize=9.2, leading=14, spaceAfter=5),
        "Bullet": ParagraphStyle("Bullet", parent=ss["BodyText"], fontName="STSong-Light", fontSize=9.1, leading=13, leftIndent=13, firstLineIndent=-8, spaceAfter=2),
        "TableCell": ParagraphStyle("TableCell", parent=ss["BodyText"], fontName="STSong-Light", fontSize=7.2, leading=9.2),
        "Caption": ParagraphStyle("Caption", parent=ss["BodyText"], fontName="STSong-Light", fontSize=8, leading=11, textColor=colors.HexColor("#555555"), alignment=TA_CENTER),
    }

    story = []
    lines = MD.read_text(encoding="utf-8").splitlines()
    i = 0
    image_re = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["H2"]))
        elif line.startswith("!["):
            m = image_re.search(line)
            if m:
                path = Path(m.group(1))
                if path.exists():
                    im = Image(str(path))
                    max_w, max_h = 175 * mm, 125 * mm
                    scale = min(max_w / im.imageWidth, max_h / im.imageHeight, 1.0)
                    im.drawWidth, im.drawHeight = im.imageWidth * scale, im.imageHeight * scale
                    story.extend([Spacer(1, 3), im, Spacer(1, 3)])
        elif line.startswith("- "):
            story.append(Paragraph("• " + inline(line[2:]), styles["Bullet"]))
        elif re.match(r"^\d+\. ", line):
            story.append(Paragraph(inline(line), styles["Bullet"]))
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            table = table_from(block, styles)
            if table:
                story.extend([Spacer(1, 3), table, Spacer(1, 7)])
            continue
        else:
            story.append(Paragraph(inline(line), styles["Body"]))
        i += 1

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("STSong-Light", 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(15 * mm, 9 * mm, "Package A2D LingBot-VLA LoRA 离线评估报告")
        canvas.drawRightString(195 * mm, 9 * mm, f"第 {doc.page} 页")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm, topMargin=14 * mm, bottomMargin=16 * mm, title="Package A2D LingBot-VLA LoRA 训练与离线评估报告", author="OpenAI")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(PDF)


if __name__ == "__main__":
    main()

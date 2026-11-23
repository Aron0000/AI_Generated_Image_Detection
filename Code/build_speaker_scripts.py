"""Generate one Word script per speaker from Report/DEFENSE_SCRIPT.md.

The markdown file is the single source of truth: the slide order, timings,
English spoken text, Chinese stage cues and handoffs are all parsed from it.
Output: Report/scripts/01_Huanrong_Li.docx ... 05_Yangqi_Zhang.docx
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "Report" / "DEFENSE_SCRIPT.md"
OUT_DIR = ROOT / "Report" / "scripts"

NAVY = RGBColor(0x0F, 0x2A, 0x44)
BLUE = RGBColor(0x25, 0x63, 0xEB)
AMBER = RGBColor(0x97, 0x5A, 0x00)
GRAY = RGBColor(0x5B, 0x66, 0x70)
LIGHT = "F1F5F9"

PART_RE = re.compile(r"^## PART (\d+) \u2014 (.+?) \(Slides (\d+)\u2013(\d+), ([0-9:]+)\u2013([0-9:]+)\)")
SLIDE_RE = re.compile(r"^### Slide (\d+) \u2014 (.+?) \(([0-9:]+)\u2013([0-9:]+)\)")
Q_RE = re.compile(r"^\*\*Q(\d+)\. (.+?)\*\*$")

SPEAKERS = {
    "Huanrong Li": ("01_Huanrong_Li.docx", [1, 4, 5, 13]),
    "Bo Pang": ("02_Bo_Pang.docx", [6, 7, 14]),
    "Huanyu Zhang": ("03_Huanyu_Zhang.docx", [3, 9, 10]),
    "Junxu Chen": ("04_Junxu_Chen.docx", [2, 12]),
    "Yangqi Zhang": ("05_Yangqi_Zhang.docx", [8, 11, 15]),
}


@dataclass
class Slide:
    number: int
    title: str
    start: str
    end: str
    on_screen: str = ""
    cue: str = ""
    english: list[str] = field(default_factory=list)
    handoff: str = ""

    @property
    def words(self) -> int:
        return sum(len(line.split()) for line in self.english)


@dataclass
class Part:
    number: int
    speaker: str
    first_slide: int
    last_slide: int
    start: str
    end: str
    slides: list[Slide] = field(default_factory=list)


def parse_parts(text: str) -> list[Part]:
    parts: list[Part] = []
    part: Part | None = None
    slide: Slide | None = None
    mode = ""
    for raw in text.splitlines():
        line = raw.strip()
        m = PART_RE.match(line)
        if m:
            part = Part(int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)),
                        m.group(5), m.group(6))
            parts.append(part)
            slide = None
            mode = ""
            continue
        m = SLIDE_RE.match(line)
        if m and part is not None:
            slide = Slide(int(m.group(1)), m.group(2), m.group(3), m.group(4))
            part.slides.append(slide)
            mode = ""
            continue
        if slide is None:
            continue
        if line.startswith("**On screen:**"):
            slide.on_screen = line.split("**On screen:**", 1)[1].strip()
            mode = ""
        elif line.startswith("**\u4e2d\u6587\u63d0\u793a\uff1a**"):
            slide.cue = line.split("**\u4e2d\u6587\u63d0\u793a\uff1a**", 1)[1].strip()
            mode = ""
        elif line == "**English:**":
            mode = "english"
        elif line.startswith("**Handoff:**"):
            slide.handoff = line.split("**Handoff:**", 1)[1].strip()
            mode = ""
        elif mode == "english" and line.startswith(">"):
            slide.english.append(line[1:].strip())
        elif line.startswith("**") and mode == "english":
            mode = ""
    return parts


def parse_questions(text: str) -> dict[int, tuple[str, str]]:
    lines = text.splitlines()
    questions: dict[int, tuple[str, str]] = {}
    i = 0
    while i < len(lines):
        m = Q_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        number, title = int(m.group(1)), m.group(2)
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        answer_lines = []
        while i < len(lines) and lines[i].strip() and not Q_RE.match(lines[i].strip()):
            answer_lines.append(lines[i].strip())
            i += 1
        questions[number] = (title, " ".join(answer_lines))
    return questions


def set_font(run, size=11, bold=False, italic=False, color=NAVY, name="Calibri"):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), "Microsoft YaHei")


def shade(paragraph, hex_fill: str):
    ppr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    ppr.append(shd)


def bottom_border(paragraph, color="2563EB", size=8):
    ppr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), "4")
    bottom.set(qn("w:color"), color)
    borders.append(bottom)
    ppr.append(borders)


def add_label(doc: Document, text: str, color=BLUE):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    set_font(run, size=8.5, bold=True, color=color)
    return p


def add_body(doc: Document, text: str, size=11, color=NAVY, italic=False, bold=False, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.3
    run = p.add_run(text)
    set_font(run, size=size, bold=bold, italic=italic, color=color)
    return p


def add_meta(paragraph, text: str, size=9.5, color=GRAY):
    run = paragraph.add_run(text)
    set_font(run, size=size, color=color)


def render_table(doc: Document, slides: list[Slide]):
    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    widths = [Cm(1.7), Cm(8.2), Cm(3.0), Cm(2.0)]
    headers = ["Slide", "Title", "Clock", "Words"]
    for cell, header, width in zip(table.rows[0].cells, headers, widths):
        cell.width = width
        paragraph = cell.paragraphs[0]
        run = paragraph.add_run(header)
        set_font(run, size=9, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
        shade(paragraph, "0F2A44")
    for slide in slides:
        row = table.add_row()
        values = [str(slide.number), slide.title, f"{slide.start}\u2013{slide.end}", str(slide.words)]
        for cell, value, width in zip(row.cells, values, widths):
            cell.width = width
            run = cell.paragraphs[0].add_run(value)
            set_font(run, size=9.5, color=NAVY)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def build(part: Part, filename: str, questions: dict[int, tuple[str, str]]):
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(1.9)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run(f"EE5438 Group 20  \u00b7  {part.speaker}  \u00b7  Slides {part.first_slide}\u2013{part.last_slide}")
    set_font(run, size=8, color=GRAY)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run("EE5438  \u00b7  GROUP 20  \u00b7  DEFENCE SCRIPT")
    set_font(run, size=9, bold=True, color=BLUE)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(part.speaker)
    set_font(run, size=24, bold=True, color=NAVY)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run(f"Slides {part.first_slide}\u2013{part.last_slide}   \u00b7   {part.start}\u2013{part.end}")
    set_font(run, size=11, color=GRAY)
    bottom_border(p)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run("Read the English aloud. Text in square brackets is a stage direction; do not read it. "
                    "The Chinese line tells you what to point at or when to pause.")
    set_font(run, size=9.5, italic=True, color=GRAY)
    p = doc.add_paragraph()
    run = p.add_run("\u4e2d\u6587\u63d0\u793a\u4e0d\u9700\u8981\u8bfb\u51fa\uff0c\u53ea\u7528\u4e8e\u63d0\u9192\u4f60\u7684\u52a8\u4f5c\u3001\u8bed\u901f\u548c\u4ea4\u63a5\u3002")
    set_font(run, size=9.5, italic=True, color=GRAY)

    add_label(doc, "RUN OF SHOW")
    render_table(doc, part.slides)

    for slide in part.slides:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(f"Slide {slide.number}  \u00b7  {slide.title}")
        set_font(run, size=15, bold=True, color=NAVY)
        bottom_border(p)

        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        add_meta(p, f"{slide.start}\u2013{slide.end}   \u00b7   {slide.words} spoken words")

        add_label(doc, "ON SCREEN")
        add_body(doc, slide.on_screen, size=9.5, color=GRAY, italic=True)

        add_label(doc, "\u4e2d\u6587\u821e\u53f0\u63d0\u793a", color=AMBER)
        add_body(doc, slide.cue, size=10.5, color=NAVY)

        add_label(doc, "SPOKEN SCRIPT")
        for line in slide.english:
            add_body(doc, line, size=12, color=NAVY)

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        run = p.add_run("Handoff:  ")
        set_font(run, size=9, bold=True, color=BLUE)
        run = p.add_run(slide.handoff)
        set_font(run, size=9.5, italic=True, color=GRAY)

    if part.speaker in SPEAKERS:
        picked = SPEAKERS[part.speaker][1]
        if picked:
            doc.add_page_break()
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run("Likely questions for your section")
            set_font(run, size=15, bold=True, color=NAVY)
            bottom_border(p)
            for number in picked:
                if number not in questions:
                    continue
                title, answer = questions[number]
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(10)
                p.paragraph_format.space_after = Pt(2)
                run = p.add_run(f"Q{number}. {title}")
                set_font(run, size=10.5, bold=True, color=NAVY)
                add_body(doc, answer, size=10, color=GRAY)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / filename
    doc.save(target)
    return target


def main():
    text = SOURCE.read_text(encoding="utf-8")
    parts = parse_parts(text)
    questions = parse_questions(text)
    if len(parts) != 5:
        raise SystemExit(f"expected 5 parts, found {len(parts)}")
    for part in parts:
        if not part.slides:
            raise SystemExit(f"part {part.number} has no slides")
        filename, _ = SPEAKERS[part.speaker]
        target = build(part, filename, questions)
        print(f"{part.speaker:16s} slides {part.first_slide:2d}-{part.last_slide:2d}  "
              f"{part.start}-{part.end}  -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

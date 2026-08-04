from io import BytesIO
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# ReportLab's built-in base-14 fonts (Helvetica, Times-Roman, ...) only cover
# a Latin-1-like subset and silently render unsupported glyphs as boxes.
# Evidence and justifications in this project can include non-English
# characters (e.g. Latvian diacritics, Bulgarian Cyrillic), so a Unicode
# TrueType font is registered here and used for every style below instead.
FONTS_DIR = Path(__file__).resolve().parent / "fonts"
UNICODE_FONT = "DejaVuSans"
UNICODE_FONT_BOLD = "DejaVuSans-Bold"

pdfmetrics.registerFont(TTFont(UNICODE_FONT, str(FONTS_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont(UNICODE_FONT_BOLD, str(FONTS_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily(
    UNICODE_FONT,
    normal=UNICODE_FONT,
    bold=UNICODE_FONT_BOLD,
    italic=UNICODE_FONT,
    boldItalic=UNICODE_FONT_BOLD,
)


def safe_text(value) -> str:
    return str(value or "").strip()


def _escape_pdf_text(text: str) -> str:
    return (
        safe_text(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def build_result_pdf_bytes(result: dict) -> bytes:
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    # Built-in styles default to Helvetica/Times, which don't cover the
    # non-English characters this project's evidence can contain. Repoint
    # every style used below at the registered Unicode font.
    for style_name in ("Title", "Heading2", "BodyText"):
        styles[style_name].fontName = UNICODE_FONT

    styles.add(
        ParagraphStyle(
            name="SmallBody",
            parent=styles["BodyText"],
            fontName=UNICODE_FONT,
            fontSize=10,
            leading=14,
            spaceAfter=6,
        )
    )

    story = []

    title = f"ODM Policy & Impact Explorer Result"
    story.append(Paragraph(_escape_pdf_text(title), styles["Title"]))
    story.append(Spacer(1, 8))

    meta_lines = [
        f"<b>Country:</b> {_escape_pdf_text(result.get('country', ''))}",
        f"<b>Question ID:</b> {_escape_pdf_text(result.get('question_id', ''))}",
        f"<b>Dimension:</b> {_escape_pdf_text(result.get('dimension', ''))}",
        f"<b>Mode:</b> {_escape_pdf_text(result.get('mode', ''))}",
        f"<b>Answer:</b> {_escape_pdf_text(result.get('final_answer_selected', ''))}",
        f"<b>Score:</b> {_escape_pdf_text(result.get('score_suggestion', ''))}",
        f"<b>Confidence:</b> {_escape_pdf_text(result.get('confidence', ''))}",
        f"<b>Evidence sufficiency:</b> {_escape_pdf_text(result.get('evidence_sufficiency', ''))}",
    ]
    story.append(Paragraph("<br/>".join(meta_lines), styles["SmallBody"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Question</b>", styles["Heading2"]))
    story.append(Paragraph(_escape_pdf_text(result.get("question", "")), styles["SmallBody"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Justification</b>", styles["Heading2"]))
    story.append(Paragraph(_escape_pdf_text(result.get("justification_generated", "")), styles["SmallBody"]))
    story.append(Spacer(1, 10))

    sources = result.get("retrieved_sources", []) or []
    story.append(Paragraph("<b>Retrieved Sources</b>", styles["Heading2"]))

    if not sources:
        story.append(Paragraph("No retrieved sources available.", styles["SmallBody"]))
    else:
        for i, src in enumerate(sources, start=1):
            source_text = (
                f"<b>[{i}] { _escape_pdf_text(src.get('source_title', 'Untitled source')) }</b><br/>"
                f"URL: {_escape_pdf_text(src.get('source_url', ''))}<br/>"
                f"Similarity: {_escape_pdf_text(src.get('similarity_score', ''))}<br/>"
                f"Snippet: {_escape_pdf_text((src.get('chunk_text', '') or '')[:1000])}"
            )
            story.append(Paragraph(source_text, styles["SmallBody"]))
            story.append(Spacer(1, 6))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


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
    styles.add(
        ParagraphStyle(
            name="SmallBody",
            parent=styles["BodyText"],
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
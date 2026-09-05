"""Generates synthetic PDF/DOCX/PPTX/XLSX fixtures with real embedded charts,
for exercising the Phase 1-3 extractors before real corpus files with
embedded charts are available. Not part of the runtime app - dev/test only
(requires matplotlib, which is intentionally not in requirements.txt).
"""

import io
from pathlib import Path

import docx
import fitz
import matplotlib
import openpyxl
from openpyxl.chart import BarChart, Reference
from pptx import Presentation
from pptx.util import Inches

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
REVENUE = [42, 58, 51, 73]


def make_chart_png() -> bytes:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(QUARTERS, REVENUE, color="steelblue")
    ax.set_title("Kestrel Quarterly Revenue ($M)")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Revenue ($M)")
    for i, v in enumerate(REVENUE):
        ax.text(i, v + 1, str(v), ha="center")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def make_pdf(chart_png: bytes) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Kestrel Financial Summary")
    page.insert_text(
        (72, 100),
        "Revenue grew steadily across all four quarters of the fiscal year.",
    )
    rect = fitz.Rect(72, 130, 400, 320)
    page.insert_image(rect, stream=chart_png)
    doc.save(FIXTURES_DIR / "sample.pdf")
    doc.close()


def make_docx(chart_png: bytes) -> None:
    d = docx.Document()
    d.add_heading("Kestrel Financial Summary", level=1)
    d.add_paragraph("Revenue grew steadily across all four quarters of the fiscal year.")

    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Quarter"
    table.rows[0].cells[1].text = "Revenue ($M)"
    for q, r in zip(QUARTERS, REVENUE):
        row = table.add_row()
        row.cells[0].text = q
        row.cells[1].text = str(r)

    chart_path = FIXTURES_DIR / "_chart_tmp.png"
    chart_path.write_bytes(chart_png)
    d.add_picture(str(chart_path))
    d.save(FIXTURES_DIR / "sample.docx")
    chart_path.unlink()


def make_pptx(chart_png: bytes) -> None:
    prs = Presentation()

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Kestrel Financial Summary"
    slide.placeholders[1].text_frame.text = (
        "Revenue grew steadily across all four quarters of the fiscal year."
    )
    slide.notes_slide.notes_text_frame.text = (
        "Speaker note: mention the Q4 uptick was driven by the Nightjar rollout."
    )

    chart_path = FIXTURES_DIR / "_chart_tmp.png"
    chart_path.write_bytes(chart_png)
    picture_slide = prs.slides.add_slide(prs.slide_layouts[5])
    picture_slide.shapes.title.text = "Revenue Chart (image)"
    picture_slide.shapes.add_picture(str(chart_path), Inches(1), Inches(1.5), width=Inches(6))
    chart_path.unlink()

    native_chart_slide = prs.slides.add_slide(prs.slide_layouts[5])
    native_chart_slide.shapes.title.text = "Revenue Chart (native)"
    chart_data = __import__("pptx.chart.data", fromlist=["CategoryChartData"]).CategoryChartData()
    chart_data.categories = QUARTERS
    chart_data.add_series("Revenue ($M)", REVENUE)
    from pptx.enum.chart import XL_CHART_TYPE

    native_chart_slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1.5), Inches(6), Inches(4), chart_data
    )

    prs.save(FIXTURES_DIR / "sample.pptx")


def make_xlsx() -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Revenue"
    ws.append(["Quarter", "Revenue ($M)"])
    for q, r in zip(QUARTERS, REVENUE):
        ws.append([q, r])

    chart = BarChart()
    chart.title = "Kestrel Quarterly Revenue"
    chart.x_axis.title = "Quarter"
    chart.y_axis.title = "Revenue ($M)"
    data = Reference(ws, min_col=2, min_row=1, max_row=5)
    cats = Reference(ws, min_col=1, min_row=2, max_row=5)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    ws.add_chart(chart, "D2")

    wb.save(FIXTURES_DIR / "sample.xlsx")


def main() -> None:
    chart_png = make_chart_png()
    make_pdf(chart_png)
    make_docx(chart_png)
    make_pptx(chart_png)
    make_xlsx()
    print("fixtures written to", FIXTURES_DIR)


if __name__ == "__main__":
    main()

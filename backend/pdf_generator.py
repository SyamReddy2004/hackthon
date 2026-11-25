"""Pure-Python standard PDF generator for Credit Fairness Audit Reports.
Produces valid PDF-1.4 documents with no external dependencies (zero C-extensions, zero network).
"""
import io
from typing import Any, Dict, List, Optional


class SimplePDFWriter:
    """Lightweight, standards-compliant PDF-1.4 builder."""

    def __init__(self, page_width: float = 612.0, page_height: float = 792.0):
        # 612 x 792 points = standard US Letter (8.5 x 11 in)
        self.w = page_width
        self.h = page_height
        self.pages: List[str] = []
        self.current_stream: List[str] = []
        self.margin_x = 45.0
        self.margin_top = 45.0
        self.cursor_y = self.h - self.margin_top

    def new_page(self) -> None:
        if self.current_stream:
            self.pages.append("\n".join(self.current_stream))
            self.current_stream = []
        self.cursor_y = self.h - self.margin_top

    def _escape_text(self, text: str) -> str:
        # Standard ASCII escape for PDF string literals
        clean = ""
        for ch in text:
            code = ord(ch)
            if ch in ("\\", "(", ")"):
                clean += "\\" + ch
            elif 32 <= code <= 126:
                clean += ch
            else:
                # Replace unsupported unicode with closest ASCII
                replacements = {
                    "–": "-", "—": "-", """: '"', """: '"',
                    "'": "'", "'": "'", "•": "*", "≥": ">=",
                    "≤": "<=", "→": "->", "←": "<-", "±": "+/-",
                }
                clean += replacements.get(ch, " ")
        return clean

    def draw_rect(self, x: float, y: float, w: float, h: float, fill_rgb: tuple = None, stroke_rgb: tuple = None, line_width: float = 1.0) -> None:
        ops = []
        if fill_rgb:
            ops.append(f"{fill_rgb[0]:.3f} {fill_rgb[1]:.3f} {fill_rgb[2]:.3f} rg")
        if stroke_rgb:
            ops.append(f"{stroke_rgb[0]:.3f} {stroke_rgb[1]:.3f} {stroke_rgb[2]:.3f} RG")
            ops.append(f"{line_width:.2f} w")
        ops.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re")
        if fill_rgb and stroke_rgb:
            ops.append("B")
        elif fill_rgb:
            ops.append("f")
        elif stroke_rgb:
            ops.append("S")
        self.current_stream.append(" ".join(ops))

    def draw_line(self, x1: float, y1: float, x2: float, y2: float, stroke_rgb: tuple = (0.7, 0.7, 0.7), line_width: float = 0.8) -> None:
        op = f"{stroke_rgb[0]:.3f} {stroke_rgb[1]:.3f} {stroke_rgb[2]:.3f} RG {line_width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S"
        self.current_stream.append(op)

    def draw_text(self, text: str, x: float, y: float, font: str = "F1", size: float = 10.0, rgb: tuple = (0.1, 0.1, 0.1)) -> None:
        safe_text = self._escape_text(text)
        op = (
            f"BT /{font} {size:.1f} Tf "
            f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} rg "
            f"1 0 0 1 {x:.2f} {y:.2f} Tm ({safe_text}) Tj ET"
        )
        self.current_stream.append(op)

    def add_wrapped_paragraph(
        self,
        text: str,
        font: str = "F1",
        size: float = 9.5,
        leading: float = 13.0,
        rgb: tuple = (0.15, 0.15, 0.15),
        max_chars: int = 90,
        indent: float = 0.0,
    ) -> None:
        words = text.split(" ")
        line = ""
        for word in words:
            test_line = line + (" " if line else "") + word
            if len(test_line) > max_chars:
                if self.cursor_y < 60:
                    self.new_page()
                self.draw_text(line, self.margin_x + indent, self.cursor_y, font=font, size=size, rgb=rgb)
                self.cursor_y -= leading
                line = word
            else:
                line = test_line
        if line:
            if self.cursor_y < 60:
                self.new_page()
            self.draw_text(line, self.margin_x + indent, self.cursor_y, font=font, size=size, rgb=rgb)
            self.cursor_y -= leading

    def build(self) -> bytes:
        if self.current_stream:
            self.pages.append("\n".join(self.current_stream))
            self.current_stream = []

        total_pages = max(1, len(self.pages))
        objs: List[str] = []

        # Object 1: Catalog
        objs.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj")

        # Object 2: Pages container
        kids = " ".join([f"{3 + i * 2} 0 R" for i in range(total_pages)])
        objs.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {total_pages} >>\nendobj")

        # Object font: F1 = Helvetica, F2 = Helvetica-Bold
        font_obj_idx = 3 + total_pages * 2
        font_objs = [
            f"{font_obj_idx} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj",
            f"{font_obj_idx + 1} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj",
        ]

        # Page and Stream objects
        for i, page_stream in enumerate(self.pages):
            page_obj_num = 3 + i * 2
            content_obj_num = page_obj_num + 1
            page_dict = (
                f"{page_obj_num} 0 obj\n"
                f"<< /Type /Page /Parent 2 0 R\n"
                f"   /MediaBox [0 0 {self.w:.1f} {self.h:.1f}]\n"
                f"   /Contents {content_obj_num} 0 R\n"
                f"   /Resources << /Font << /F1 {font_obj_idx} 0 R /F2 {font_obj_idx + 1} 0 R >> >>\n"
                f">>\nendobj"
            )
            objs.append(page_dict)

            stream_bytes = page_stream.encode("latin-1", "replace")
            content_dict = (
                f"{content_obj_num} 0 obj\n"
                f"<< /Length {len(stream_bytes)} >>\n"
                f"stream\n"
                f"{page_stream}\n"
                f"endstream\nendobj"
            )
            objs.append(content_dict)

        objs.extend(font_objs)

        # Assemble PDF with cross-reference table
        output = io.BytesIO()
        output.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        offsets = []
        for obj_str in objs:
            offsets.append(output.tell())
            output.write(obj_str.encode("latin-1", "replace"))
            output.write(b"\n")

        xref_offset = output.tell()
        output.write(b"xref\n")
        output.write(f"0 {len(objs) + 1}\n".encode("ascii"))
        output.write(b"0000000000 65535 f \n")
        for off in offsets:
            output.write(f"{off:010d} 00000 n \n".encode("ascii"))

        trailer = (
            f"trailer\n"
            f"<< /Size {len(objs) + 1}\n"
            f"   /Root 1 0 R\n"
            f">>\n"
            f"startxref\n"
            f"{xref_offset}\n"
            f"%%EOF\n"
        )
        output.write(trailer.encode("ascii"))
        return output.getvalue()


def create_audit_report_pdf(
    overview: Dict[str, Any],
    gender_fairness: Dict[str, Any],
    age_fairness: Dict[str, Any],
    applicant: Dict[str, Any],
    row_index: int = 0,
) -> bytes:
    """Generates an executive-ready Credit Fairness Audit Report in PDF format."""
    pdf = SimplePDFWriter()

    # HEADER & TITLE BAR
    pdf.draw_rect(45, 735, 522, 38, fill_rgb=(0.08, 0.28, 0.65))
    pdf.draw_text("PS173 CREDIT FAIRNESS & DECISION EXPLAINABILITY AUDIT", 55, 750, font="F2", size=13, rgb=(1, 1, 1))
    pdf.draw_text("Official Human-in-the-Loop Risk Assessment Document", 55, 740, font="F1", size=8.5, rgb=(0.85, 0.9, 1))

    pdf.cursor_y = 715

    # POLICY GUARDRAIL BOX
    pdf.draw_rect(45, pdf.cursor_y - 28, 522, 36, fill_rgb=(1.0, 0.97, 0.88), stroke_rgb=(0.95, 0.65, 0.15), line_width=1.0)
    pdf.draw_text("MANDATORY REGULATORY GUARDRAIL:", 55, pdf.cursor_y - 12, font="F2", size=8.5, rgb=(0.6, 0.3, 0.0))
    pdf.draw_text("Automated lending rejections are PROHIBITED. All model flags are advisory risk signals", 55, pdf.cursor_y - 22, font="F1", size=8.5, rgb=(0.5, 0.25, 0.0))
    pdf.draw_text("requiring credit officer assessment based solely on financial repayment capacity.", 55, pdf.cursor_y - 32, font="F1", size=8.5, rgb=(0.5, 0.25, 0.0))
    pdf.cursor_y -= 52

    # 1. PORTFOLIO METRICS
    pdf.draw_text("1. PORTFOLIO FAIRNESS & RISK OVERVIEW", 45, pdf.cursor_y, font="F2", size=10.5, rgb=(0.1, 0.2, 0.4))
    pdf.draw_line(45, pdf.cursor_y - 3, 567, pdf.cursor_y - 3, stroke_rgb=(0.8, 0.85, 0.9))
    pdf.cursor_y -= 16

    col_w = 125
    metrics = [
        ("Audited Applications", f"{overview.get('applications_audited', 0):,}"),
        ("Actual Default Rate", f"{overview.get('actual_default_rate_pct', 0)}%"),
        ("Flagged for Review", f"{overview.get('flagged_for_review_pct', 0)}%"),
        ("Mean Risk Score", f"{overview.get('mean_risk_score_pct', 0)}%"),
    ]
    for idx, (label, val) in enumerate(metrics):
        bx = 45 + idx * (col_w + 7)
        pdf.draw_rect(bx, pdf.cursor_y - 22, col_w, 28, fill_rgb=(0.96, 0.97, 0.98), stroke_rgb=(0.85, 0.88, 0.92))
        pdf.draw_text(label.upper(), bx + 6, pdf.cursor_y - 3, font="F1", size=7.0, rgb=(0.4, 0.45, 0.5))
        pdf.draw_text(val, bx + 6, pdf.cursor_y - 17, font="F2", size=11.0, rgb=(0.1, 0.15, 0.2))
    pdf.cursor_y -= 38

    # 2. GROUP FAIRNESS METRICS
    pdf.draw_text("2. DEMOGRAPHIC GROUP DISPARITY ANALYSIS", 45, pdf.cursor_y, font="F2", size=10.5, rgb=(0.1, 0.2, 0.4))
    pdf.draw_line(45, pdf.cursor_y - 3, 567, pdf.cursor_y - 3, stroke_rgb=(0.8, 0.85, 0.9))
    pdf.cursor_y -= 14

    for grp_data in (gender_fairness, age_fairness):
        grp_name = grp_data.get("group_by", "DEMOGRAPHIC")
        pdf.draw_text(f"Demographic Dimension: {grp_name}", 45, pdf.cursor_y, font="F2", size=9, rgb=(0.2, 0.25, 0.35))
        pdf.cursor_y -= 11

        takeaway = grp_data.get("takeaway", "")
        if takeaway:
            pdf.add_wrapped_paragraph(takeaway, font="F1", size=8.0, leading=10.5, rgb=(0.3, 0.35, 0.4), max_chars=105)

        # Table header
        headers = [("Group", 70), ("Applicants", 75), ("Flagged (%)", 75), ("Actual Default (%)", 85), ("False-Positive (%)", 85)]
        tx = 45
        pdf.draw_rect(45, pdf.cursor_y - 12, 522, 14, fill_rgb=(0.92, 0.94, 0.97))
        for h_label, h_w in headers:
            pdf.draw_text(h_label, tx + 4, pdf.cursor_y - 9, font="F2", size=7.5, rgb=(0.2, 0.25, 0.3))
            tx += h_w
        pdf.cursor_y -= 15

        for g in grp_data.get("groups", [])[:4]:
            tx = 45
            pdf.draw_rect(45, pdf.cursor_y - 10, 522, 12, fill_rgb=(0.99, 0.99, 1.0))
            row_vals = [
                str(g.get("group", "")),
                f"{g.get('applicants', 0):,}",
                f"{g.get('flagged_pct', 0)}%",
                f"{g.get('actual_default_pct', 0)}%",
                f"{g.get('false_positive_rate_pct', 0)}%",
            ]
            for val, (_, h_w) in zip(row_vals, headers):
                pdf.draw_text(val, tx + 4, pdf.cursor_y - 8, font="F1", size=7.5, rgb=(0.2, 0.2, 0.2))
                tx += h_w
            pdf.cursor_y -= 12
        pdf.cursor_y -= 6

    # 3. INDIVIDUAL APPLICANT DECISION RATIONALE
    if pdf.cursor_y < 280:
        pdf.new_page()

    applicant_title = f"3. APPLICANT RATIONALE & RISK ASSESSMENT (TEST ROW #{row_index})"
    pdf.draw_text(applicant_title, 45, pdf.cursor_y, font="F2", size=10.5, rgb=(0.1, 0.2, 0.4))
    pdf.draw_line(45, pdf.cursor_y - 3, 567, pdf.cursor_y - 3, stroke_rgb=(0.8, 0.85, 0.9))
    pdf.cursor_y -= 16

    # Summary box
    risk_pct = applicant.get("risk_percentage", 0)
    risk_tier = applicant.get("risk_tier", "Moderate Risk")
    advisory_dir = applicant.get("advisory_direction", "Review")
    hist_outcome = applicant.get("historical_outcome", "N/A")

    pdf.draw_rect(45, pdf.cursor_y - 24, 522, 28, fill_rgb=(0.97, 0.98, 0.99), stroke_rgb=(0.85, 0.88, 0.92))
    pdf.draw_text(f"Predicted Risk Score: {risk_pct}% ({risk_tier})", 55, pdf.cursor_y - 9, font="F2", size=9, rgb=(0.15, 0.2, 0.3))
    pdf.draw_text(f"Advisory Direction: {advisory_dir}", 230, pdf.cursor_y - 9, font="F2", size=9, rgb=(0.15, 0.2, 0.3))
    pdf.draw_text(f"Historical Outcome: {hist_outcome}", 420, pdf.cursor_y - 9, font="F1", size=8.5, rgb=(0.3, 0.35, 0.4))
    pdf.cursor_y -= 34

    # Financial details strip
    feats = applicant.get("applicant_features", {})
    details_str = (
        f"Income: {feats.get('AMT_INCOME_TOTAL', 'N/A')} | "
        f"Credit: {feats.get('AMT_CREDIT', 'N/A')} | "
        f"Annuity: {feats.get('AMT_ANNUITY', 'N/A')} | "
        f"EXT_2: {feats.get('EXT_SOURCE_2', 'N/A')} | "
        f"EXT_3: {feats.get('EXT_SOURCE_3', 'N/A')}"
    )
    pdf.draw_text(f"Verified Financials: {details_str}", 45, pdf.cursor_y, font="F1", size=8.0, rgb=(0.35, 0.4, 0.45))
    pdf.cursor_y -= 14

    # Why it leans toward acceptance
    pdf.draw_text("Factors Supporting Loan Acceptance (Lower Risk):", 45, pdf.cursor_y, font="F2", size=8.5, rgb=(0.1, 0.5, 0.2))
    pdf.cursor_y -= 11
    acc_drivers = applicant.get("acceptance_drivers", [])
    if acc_drivers:
        for d in acc_drivers[:3]:
            pdf.add_wrapped_paragraph(f"- {d.get('summary', '')}", font="F1", size=8.0, leading=10.5, rgb=(0.15, 0.4, 0.2), indent=10, max_chars=105)
    else:
        pdf.draw_text("- No strong risk-reducing factors identified.", 55, pdf.cursor_y, font="F1", size=8.0, rgb=(0.4, 0.4, 0.4))
        pdf.cursor_y -= 11

    # Why it leans toward rejection
    pdf.draw_text("Factors Raising Repayment Concern (Higher Risk):", 45, pdf.cursor_y, font="F2", size=8.5, rgb=(0.7, 0.15, 0.15))
    pdf.cursor_y -= 11
    rej_drivers = applicant.get("rejection_drivers", [])
    if rej_drivers:
        for d in rej_drivers[:3]:
            pdf.add_wrapped_paragraph(f"- {d.get('summary', '')}", font="F1", size=8.0, leading=10.5, rgb=(0.6, 0.15, 0.15), indent=10, max_chars=105)
    else:
        pdf.draw_text("- No critical repayment red flags identified.", 55, pdf.cursor_y, font="F1", size=8.0, rgb=(0.4, 0.4, 0.4))
        pdf.cursor_y -= 11

    # 4. FAIRNESS ALERT / BIAS MITIGATION
    alerts = applicant.get("fairness_alerts", [])
    if alerts:
        if pdf.cursor_y < 120:
            pdf.new_page()
        for alert in alerts:
            box_h = 42
            pdf.draw_rect(45, pdf.cursor_y - box_h + 4, 522, box_h, fill_rgb=(1.0, 0.98, 0.92), stroke_rgb=(0.9, 0.65, 0.2), line_width=1.0)
            pdf.draw_text(f"FAIRNESS ALERT: {alert.get('title', 'Protected Attribute Disparity')}", 55, pdf.cursor_y - 7, font="F2", size=8.0, rgb=(0.75, 0.35, 0.0))
            pdf.draw_text(f"Finding: {alert.get('finding', '')}", 55, pdf.cursor_y - 17, font="F1", size=7.5, rgb=(0.45, 0.25, 0.05))
            pdf.draw_text(f"Directive: {alert.get('directive', '')}", 55, pdf.cursor_y - 27, font="F2", size=7.5, rgb=(0.2, 0.2, 0.2))
            pdf.cursor_y -= box_h + 4

    # 5. HUMAN SIGN-OFF BLOCK
    if pdf.cursor_y < 80:
        pdf.new_page()

    pdf.cursor_y -= 8
    pdf.draw_line(45, pdf.cursor_y, 567, pdf.cursor_y, stroke_rgb=(0.8, 0.8, 0.8))
    pdf.cursor_y -= 14

    pdf.draw_text("CREDIT OFFICER FINAL HUMAN DECISION SIGN-OFF", 45, pdf.cursor_y, font="F2", size=8.5, rgb=(0.2, 0.25, 0.35))
    pdf.draw_text("Decision: [   ] Approved   [   ] Held for Verification   [   ] Rejected", 45, pdf.cursor_y - 14, font="F1", size=8.0, rgb=(0.2, 0.2, 0.2))
    pdf.draw_text("Officer Signature: _______________________      Date: ____________      Audit Token: PS173-VAL", 45, pdf.cursor_y - 28, font="F1", size=7.5, rgb=(0.4, 0.45, 0.5))

    return pdf.build()

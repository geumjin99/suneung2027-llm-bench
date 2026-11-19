"""Column-aware text extraction for two-column KICE exam PDFs (PyMuPDF)."""
import sys, fitz, re
pdf, first, last, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
doc = fitz.open(pdf)
lines_out = []
for pno in range(first - 1, last):
    page = doc[pno]
    W, H = page.rect.width, page.rect.height
    mid = W / 2
    lines = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            txt = "".join(s["text"] for s in l["spans"]).rstrip()
            if not txt.strip():
                continue
            x0, y0, x1, y1 = l["bbox"]
            if y0 < H * 0.075 or y0 > H * 0.94:   # header/footer
                continue
            col = 0 if x0 < mid - 5 else 1
            lines.append((col, round(y0, 1), x0, txt))
    lines.sort()
    lines_out.append(f"\n<<<PAGE {pno+1}>>>")
    for col, y, x, t in lines:
        lines_out.append(t)
open(out, "w").write("\n".join(lines_out))

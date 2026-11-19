"""Build per-question items for the English section.

Reading (18-45): context = the whole question set the item belongs to (same as Korean).
Listening (1-17): context = official listening script for the item + the item's printed stem and choices.
Pages whose content is mostly images (picture, chart, posters) use the VLM transcription when available.
"""
import json, os, re, sys, unicodedata
import fitz

sys.path.insert(0, os.path.dirname(__file__))
from answer_keys import english_key

EXAM = "data/pdf/27DaeSuneung_3.pdf"
SCRIPT = "data/pdf/27DaeSuneung_31.pdf"
VLM = "data/text/english_transcription_raw.json"
VLM_PAGES = (1, 2, 4)


def norm(t):
    # expand ligature glyphs only (NFKC would also turn ① into 1)
    for a, b in (("\ufb00", "ff"), ("\ufb01", "fi"), ("\ufb02", "fl"), ("\ufb03", "ffi"), ("\ufb04", "ffl")):
        t = t.replace(a, b)
    # ligature glyphs split as "f i", "f l", "f f" at the start of a word
    t = re.sub(r"\bf ([ilf])(?=[a-z])", r"f\1", t)
    return t


def column_text(page):
    W, H = page.rect.width, page.rect.height
    lines = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            txt = "".join(s["text"] for s in l["spans"]).rstrip()
            x0, y0 = l["bbox"][:2]
            if not txt.strip() or y0 < H * 0.075 or y0 > H * 0.94:
                continue
            lines.append((0 if x0 < W / 2 - 5 else 1, round(y0, 1), x0, txt))
    lines.sort()
    return "\n".join(t for *_, t in lines)


JUNK = re.compile(r"^(\d{1,2}|-\s*\d+\s*-|홀수형|영어 영역|2027학년도 대학수학능력시험 문제지|제\s*3\s*교시|이 문제지에 관한 저작권은.*|\* 확인 사항.*)$")


def clean(text):
    out, skip = [], False
    for l in text.split("\n"):
        s = l.strip()
        if s.startswith(("* 확인 사항", "※ 확인 사항", "＊ 확인 사항")):
            skip = True
        if skip or JUNK.match(s):
            continue
        out.append(s)
    return out


def main():
    doc = fitz.open(EXAM)
    pages = {p: norm(column_text(doc[p - 1])) for p in range(1, 9)}
    if os.path.exists(VLM):
        v = json.load(open(VLM))
        for p in VLM_PAGES:
            k = f"english_p{p:02d}.png"
            if k in v:
                pages[p] = norm(re.sub(r"\[(\d+)\s*[~∼]\s*(\d+)\]", r"[\1～\2]", v[k]))
    else:
        print("WARNING: VLM transcription missing; image content on pages", VLM_PAGES, "is not included")
    # picture for item 4, re-described after checking the page image
    pages[1] = re.sub(r"\[그림: 거실 내부[^\]]*\]",
        "[그림: 거실 내부. ① 별 무늬가 있는 똑같은 소파 두 개가 나란히 놓여 있다. ② 오른쪽 벽에 별 모양 시계가 걸려 있다. "
        "③ 창문에 하트 무늬 블라인드가 내려져 있다. ④ 왼쪽 벽에 다섯 칸짜리 책장이 있다. ⑤ 왼쪽 작은 탁자 위에 찻주전자가 있다. "
        "오른쪽 벽에는 번호가 붙지 않은 거울도 걸려 있다.]", pages[1])
    lines = clean("\n".join(pages[p] for p in range(1, 9)))

    # question starts (sequential numbering) and set headers
    expected, qstart, sets = 1, {}, []
    for i, l in enumerate(lines):
        m = re.match(r"^\[(\d+)～(\d+)\]", l)
        if m:
            sets.append((i, int(m.group(1)), int(m.group(2))))
            continue
        m = re.match(r"^(\d+)\.(\s|$)", l)
        if m and int(m.group(1)) == expected:
            qstart[expected] = i
            expected += 1
    assert expected == 46, f"found questions up to {expected - 1}"
    bounds = sorted([i for i, _, _ in sets] + list(qstart.values())) + [len(lines)]

    def q_end(q):
        return min(b for b in bounds if b > qstart[q])

    def block(q):
        st = next(((i, a, b) for i, a, b in sets if a <= q <= b), None)
        if st:
            i0, a, b = st
            return "\n".join(lines[i0:q_end(b)])
        return "\n".join(lines[qstart[q]:q_end(q)])

    sdoc = fitz.open(SCRIPT)
    script = {}
    for p in range(len(sdoc)):
        t = norm(sdoc[p].get_text())
        t = "\n".join(s.strip() for s in t.split("\n") if s.strip() and not JUNK.match(s.strip())
                      and "영어 영역 듣기평가 대본" not in s and "대학수학능력시험" not in s)
        script[p + 1] = t  # the script PDF has exactly one page per listening item (page 16 = set 16-17)
    script[16] = script.get(16, "") + "\n" + script.get(17, "")
    script[17] = script[16]

    key = english_key()
    items = []
    for q in range(1, 46):
        ans, pts = key[q]
        if q <= 17:
            ctx = (f"[듣기 대본] (실제 시험에서는 음성으로 들려준다)\n{script.get(q, '').strip()}\n\n"
                   f"[문제지]\n{block(q)}")
        else:
            ctx = block(q)
        items.append({"id": f"eng-all-{q:02d}", "subject": "english", "section": "listening" if q <= 17 else "reading",
                      "number": q, "type": "mcq", "points": pts, "answer": ans, "context": ctx})
    json.dump(items, open("data/english.json", "w"), ensure_ascii=False, indent=1)
    print(len(items), "items; points", sum(i["points"] for i in items), "; script items", sorted(k for k in script if k <= 17))


if __name__ == "__main__":
    main()

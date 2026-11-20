"""Build per-question items for Korean history and the text-centred social studies subjects.

Source: VLM transcription of each page (tables, graphs and pictures described in text).
Every item is self-contained, so context = that question's own block (stem, sources, choices).
"""
import json, os, re, sys
import fitz

sys.path.insert(0, os.path.dirname(__file__))
from answer_keys import single_key

VLM = "data/text/social_transcription_raw.json"
# key: (pdf code, subject field, Korean area name, output file)
SUBJECTS = {
    "history":   ("41",   "history", "한국사",               "data/history.json"),
    "ethics":    ("42_1", "social",  "사회탐구(생활과 윤리)", "data/social.json"),
    "thought":   ("42_2", "social",  "사회탐구(윤리와 사상)", "data/social.json"),
    "economics": ("42_7", "social",  "사회탐구(경제)",       "data/social.json"),
    "politics":  ("42_8", "social",  "사회탐구(정치와 법)",   "data/social.json"),
    "society":   ("42_9", "social",  "사회탐구(사회·문화)",   "data/social.json"),
}
JUNK = re.compile(r"^(\d{1,2}|\d{1,2}\s+\d{1,2}|\d\s*/\s*\d|홀수형|제\s*4\s*교시|한국사 영역|사회탐구 영역.*|2027학년도 대학수학능력시험.*|이 문제지에 관한 저작권은.*|\* 확인 사항.*|※ 확인 사항.*)$")


SET = re.compile(r"^\s*(?:\*\*)?\[(\d+)\s*[~∼～]\s*(\d+)\]")
CIRCLED, LETTERED = "①②③④⑤", "㉠㉡㉢㉣㉤"


# lines the automatic correction could not align, checked against the page image / PDF text
FIXES = {
    "history-17": [("대한민국 국제가 발표되었다", "대한국 국제가 반포되었다")],
    "economics-08": [("㉠ 저축한 금액제로", "㉠ 저렴한 요금제로")],
    "economics-15": [("| 0 | ㉠ | 200 |", "| 0 | ㉡ | 200 |"), ("ㄱ. ㉠은 ㉠의 3배", "ㄱ. ㉠은 ㉡의 3배")],
    "politics-11": [("보장的前提가", "보장의 전제가")],
    "society-14": [("① 고민 중인 여동생", "㉠ 고민 중인 여동생"), ("② 고민 중인 후배", "㉡ 고민 중인 후배"),
                   ("㉡는 모두", "㉡은 모두"), ("베노 다이어그램", "벤 다이어그램")],
}


def split_items(text, n=20):
    """Split a subject's text into question blocks; items in a set ([9～10]) get the set's shared material."""
    lines = [l.rstrip() for l in text.split("\n")]
    lines = [l for l in lines if not JUNK.match(l.strip().strip("*# "))]
    expected, start, sets = 1, {}, []
    for i, l in enumerate(lines):
        m = SET.match(l)
        if m:
            sets.append((i, int(m.group(1)), int(m.group(2))))
            continue
        # a question starts with its number and its stem ends with "?" (numbered lists inside sources do not)
        m = re.match(r"^\s*(?:\*\*|#+\s*)?(\d{1,2})\s*\.(?:\*\*)?(\s|$)", l)
        if m and int(m.group(1)) == expected and "?" in l:
            start[expected] = i
            expected += 1
    assert expected == n + 1, f"found questions up to {expected - 1}"
    bounds = sorted([i for i, _, _ in sets] + list(start.values())) + [len(lines)]
    end = lambda q: min(b for b in bounds if b > start[q])
    out = {}
    for q in range(1, n + 1):
        st = next(((i, a, b) for i, a, b in sets if a <= q <= b), None)
        i0 = st[0] if st else start[q]
        out[q] = "\n".join(lines[i0:end(st[2] if st else q)]).strip()
    return out


def pdf_items(pdf, n=20):
    """Plain PDF text per question, used only to check which ㉠-type markers each question really has."""
    t = "".join(pg.get_text() for pg in fitz.open(pdf))
    pos, q = [], 1
    for m in re.finditer(r"\n(\d{1,2})\. ", t):
        if int(m.group(1)) == q:
            pos.append(m.start()); q += 1
        if q > n:
            break
    pos.append(len(t))
    return {i + 1: t[pos[i]:pos[i + 1]] for i in range(n)}


def fix_markers(ctx, pdf_text):
    """The VLM sometimes reads ㉠/㉡ as ①/②. Restore them where the PDF text has the lettered marker."""
    have = [c for c in LETTERED if c in pdf_text]
    if not have:
        return ctx
    ctx = ctx.replace("⑦", "㉠")  # ㉠ read as ⑦ (no question has a seventh choice)
    lines = ctx.split("\n")
    choice0 = max(i for i, l in enumerate(lines) if l.lstrip().startswith("①"))
    out = []
    for i, l in enumerate(lines):
        for a, b in zip(CIRCLED, LETTERED):
            if b not in have:
                continue
            if i < choice0 and not l.lstrip().startswith(a):
                l = re.sub(a + r"(?!\s*[ㄱ-ㄹ](?:[,\s]|$))", b, l)  # body text: a reference mark unless "① ㄱ ② ㄷ"
            else:
                l = re.sub(a + r"(?=[가-힣,·])", b, l)  # in choices only "①에", "①과" etc. are references
        out.append(l)
    return "\n".join(out)


import difflib

PDF_CHARS = str.maketrans({"\u2024": "·", "ㆍ": "·", "＋": "+", "—": "-", "～": "~", "∼": "~", "◦": "○"})


def _norm(t):
    """Text without spaces plus, for each kept character, its index in the original string."""
    keep = [(c, i) for i, c in enumerate(t) if not c.isspace()]
    return "".join(c for c, _ in keep), [i for _, i in keep]


def correct_line(line, R, min_ratio=0.85):
    """Fix VLM misreadings in one line using the PDF text layer R (spaces removed).

    The line keeps the VLM's spacing and layout; characters are replaced only where the line aligns with
    the PDF text with high similarity. Lines that come from images (no PDF text) stay unchanged."""
    v, pos = _norm(line)
    if len(v) < 6 or "|" in v:
        return line
    sm = difflib.SequenceMatcher(None, R, v, autojunk=False)
    a, b, size = sm.find_longest_match(0, len(R), 0, len(v))
    if size < 4:
        return line
    s0 = max(0, a - b - 3)
    win = R[s0:a - b + len(v) + 3]
    m = difflib.SequenceMatcher(None, v, win, autojunk=False)
    blocks = [bl for bl in m.get_matching_blocks() if bl.size]
    if not blocks or m.ratio() * (len(v) + len(win)) / (2 * len(v)) < min_ratio:
        return line
    win = win[blocks[0].b - blocks[0].a if blocks[0].b >= blocks[0].a else 0:]  # align the starts
    m = difflib.SequenceMatcher(None, v, win, autojunk=False)
    ops = [o for o in m.get_opcodes()]
    while ops and ops[-1][0] in ("insert", "delete") and ops[-1][1] == len(v) or (ops and ops[-1][0] == "insert"):
        ops.pop()  # trailing PDF text belongs to the next line
    matched = sum(i2 - i1 for t, i1, i2, _, _ in ops if t == "equal")
    if matched / len(v) < min_ratio:
        return line
    out = line
    for t, i1, i2, j1, j2 in reversed(ops):
        if t == "equal":
            continue
        if i1 == i2:  # insertion: put it right after the previous kept character
            if i1 == 0:
                continue
            at = pos[i1 - 1] + 1
            out = out[:at] + win[j1:j2] + out[at:]
        else:
            out = out[:pos[i1]] + win[j1:j2] + out[pos[i2 - 1] + 1:]
    return out


def correct_text(ctx, pdf_text):
    R = _norm(pdf_text.translate(PDF_CHARS))[0]
    return "\n".join(correct_line(l, R) for l in ctx.split("\n"))


def set_points(ctx, q, pts):
    """Take the [3점] tag from the answer key (the VLM sometimes misplaces it) and put it on the stem line."""
    lines = ctx.replace(" [3점]", "").replace("[3점]", "").split("\n")
    if pts == 3:
        i = next(i for i, l in enumerate(lines) if re.match(rf"^\s*(?:\*\*|#+\s*)?{q}\s*\.", l))
        lines[i] += " [3점]"
    return "\n".join(lines)


def main():
    vlm = json.load(open(VLM))
    out = {}
    for key, (code, subject, area, path) in SUBJECTS.items():
        text = "\n".join(vlm[k] for k in sorted(vlm) if k.startswith(key + "_p"))
        blocks = split_items(text)
        ans = single_key(f"data/pdf/27DaeSuneung_{code}a.pdf")
        ref = pdf_items(f"data/pdf/27DaeSuneung_{code}.pdf")
        for q in range(1, 21):
            a, pts = ans[q]
            # a set's shared material is printed before its first question, i.e. inside the previous item's PDF text
            src = "".join(ref[i] for i in range(max(1, q - 2), q + 2) if i in ref) if blocks[q].startswith("[") else ref[q]
            ctx = set_points(fix_markers(correct_text(blocks[q], src), src), q, pts)
            for a_, b_ in FIXES.get(f"{key}-{q:02d}", []):
                assert a_ in ctx, (key, q, a_)
                ctx = ctx.replace(a_, b_)
            assert all(c in ctx for c in "①②③④⑤"), f"{key} {q}: choices missing"
            out.setdefault(path, []).append({
                "id": f"{key}-{q:02d}", "subject": subject, "section": key, "area": area,
                "number": q, "type": "mcq", "points": pts, "answer": a, "context": ctx})
    for path, items in out.items():
        json.dump(items, open(path, "w"), ensure_ascii=False, indent=1)
        print(path, len(items), "items; points", sum(i["points"] for i in items))


if __name__ == "__main__":
    main()

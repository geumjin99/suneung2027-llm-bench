"""Build per-question items for the Korean section from column-ordered text + answer key."""
import re, json, sys
raw = open("data/text/korean_raw.txt").read()
# private-use glyphs in the KICE font: book-title brackets and boxed slide numbers
raw = raw.replace("\U000f0854", "「").replace("\U000f0855", "」").replace("\U000f02b1", "[1] ").replace("\U000f02b2", "[2] ")
pages = re.split(r"\n<<<PAGE (\d+)>>>\n", raw)
# pages = ['', '1', text, '2', text, ...]
page_text = {int(pages[i]): pages[i+1] for i in range(1, len(pages), 2)}
PYMUPDF_P18 = page_text[18]
# pages whose screenshots/charts are images: use the VLM transcription (Qwen3.8-27B-FP8) instead
VLM_PAGES = (13, 15, 16, 18, 19, 20)
vlm = json.load(open("data/text/korean_transcription_raw.json"))
for p in VLM_PAGES:
    t = vlm[f"korean_p{p:02d}.png"]
    t = re.sub(r"\[(\d+)\s*[~∼]\s*(\d+)\]", r"[\1～\2]", t)
    if p == 20:  # right-aligned chat bubbles were transcribed as "text 윤호": restore "윤호: text"
        fixed = []
        for l in t.split("\n"):
            if re.search(r"\s윤호$", l) and ":" not in l[:6]:
                l = "윤호: " + re.sub(r"\s윤호$", "", l)
            elif " 윤호 " in l and ":" not in l[:6] and not l.startswith(("①", "②", "③", "④", "⑤")):
                l = "윤호: " + l.replace(" 윤호 ", " ")
            fixed.append(l)
        t = "\n".join(fixed).replace("윤호: 남이 보라, 유정, 태훈 남을", "윤호 님이 보라, 유정, 태훈 님을")\
            .replace("윤호: [공지] 이메일 주소:", "[공지] 윤호 이메일 주소:").replace("응, 각자 생각이란", "응, 각자 생각이랑")\
            .replace("'도시 관광'이라고 적힌 표지판이 있다", "'도심 관광'이라고 적힌 이름표가 붙어 있다")\
            .replace("☆☆산 729 11.8%", "☆☆산 729 11.8%\n(표에서 '전년도 대비 증가율' 열은 굵은 테두리로 둘러싸여 있음)")
    page_text[p] = t
JUNK = re.compile(r"^(\d{1,2}|\d+\s*국어 영역.*|국어 영역.*|홀수형|\((화법과 작문|언어와 매체)\)|2027학년도 대학수학능력시험 문제지|제\s*1\s*교시|국어 영역|이 문제지에 관한 저작권은.*)$")
def clean(lines):
    out = []
    skip = False
    for l in lines:
        s = l.strip()
        if s.startswith("* 확인 사항") or s.startswith("＊ 확인 사항") or s.startswith("※ 확인 사항"):
            skip = True
        if skip: continue
        if JUNK.match(s): continue
        out.append(s)
    return out

SECTIONS = {"common": range(1, 13), "hwajak": range(13, 17), "eonmae": range(17, 21)}
QRANGE = {"common": (1, 34), "hwajak": (35, 45), "eonmae": (35, 45)}

# answer key (홀수형): parse from pdftotext of 1a page 1
import subprocess
key_txt = subprocess.run(["pdftotext", "-layout", "-f", "1", "-l", "1", "data/pdf/27DaeSuneung_1a.pdf", "-"], capture_output=True, text=True).stdout
CIR = "①②③④⑤"
ans = {"common": {}, "hwajak": {}, "eonmae": {}}
for line in key_txt.splitlines():
    toks = re.findall(r"(\d+)\s+([①-⑤])\s+(\d)", line)
    for i, (q, a, p) in enumerate(toks):
        q = int(q); rec = (CIR.index(a) + 1, int(p))
        if q <= 34: ans["common"][q] = rec
        elif i == len(toks) - 2 or (len(toks) == 4 and i == 2): ans["hwajak"][q] = rec
        else: ans["eonmae"][q] = rec
# fix assignment: in rows with 4 tokens, order is common, common, hwajak, eonmae
ans = {"common": {}, "hwajak": {}, "eonmae": {}}
for line in key_txt.splitlines():
    toks = re.findall(r"(\d+)\s+([①-⑤])\s+(\d)", line)
    if len(toks) == 4:
        c1, c2, h, e = toks
        for q, a, p in (c1, c2): ans["common"][int(q)] = (CIR.index(a)+1, int(p))
        ans["hwajak"][int(h[0])] = (CIR.index(h[1])+1, int(h[2]))
        ans["eonmae"][int(e[0])] = (CIR.index(e[1])+1, int(e[2]))
    elif len(toks) == 2:
        for q, a, p in toks: ans["common"][int(q)] = (CIR.index(a)+1, int(p))
assert len(ans["common"]) == 34 and len(ans["hwajak"]) == 11 and len(ans["eonmae"]) == 11, {k: len(v) for k, v in ans.items()}

items = []
for sec, prange in SECTIONS.items():
    lines = clean(sum((page_text[p].split("\n") for p in prange), []))
    lo, hi = QRANGE[sec]
    # mark set headers and question starts (sequential numbering)
    expected = lo
    qstart = {}
    sets = []  # (line_idx, a, b)
    for i, l in enumerate(lines):
        m = re.match(r"^\[(\d+)～(\d+)\]", l)
        if m: sets.append((i, int(m.group(1)), int(m.group(2)))); continue
        m = re.match(r"^(\d+)\.(\s|$)", l)
        if m and int(m.group(1)) == expected:
            qstart[expected] = i; expected += 1
    assert expected == hi + 1, (sec, expected)
    bounds = sorted([i for i, _, _ in sets] + list(qstart.values())) + [len(lines)]
    def q_end(q):
        s = qstart[q]; return min(b for b in bounds if b > s)
    for q in range(lo, hi + 1):
        st = next(((i, a, b) for i, a, b in sets if a <= q <= b), None)
        if st:
            i0, a, b = st
            ctx = "\n".join(lines[i0:q_end(b)])
            # a later set header may precede the end if page interleaves; cut at next set header
            nxt = [i for i, _, _ in sets if i > i0]
            if nxt and nxt[0] < q_end(b):
                ctx = "\n".join(lines[i0:nxt[0]] + lines[qstart[a] if qstart[a] > nxt[0] else nxt[0]:q_end(b)]) if False else ctx
        else:
            ctx = "\n".join(lines[qstart[q]:q_end(q)])
        a_, p_ = ans[sec][q]
        items.append({"id": f"kor-{sec}-{q:02d}", "subject": "korean", "section": sec, "number": q,
                      "type": "mcq", "points": p_, "answer": a_, "context": ctx})
MK = {"sa": "\u1109\u119e\u1140\u1175", "teuk": "\u1110\u119e\u11a8"}
q39 = PYMUPDF_P18[PYMUPDF_P18.index("39. <보기>"):PYMUPDF_P18.index("[40～43]") if "[40～43]" in PYMUPDF_P18 else None]
q39 = "\n".join(clean(q39.split("\n")))
q39 = re.sub(r"<자 료>.*?\(단,", "<자 료>\n◦15세기 국어 단어의 변화\n고개 > 고개   거싀 > 거의   " + MK["sa"] + " > 사이   귀 > 귀   " + MK["teuk"] + " > 턱\n(단,", q39, flags=re.S)
for a, b in (("‘거\u200b> 거의’", "‘거싀 > 거의’"),):
    q39 = q39.replace(a, b)
q39 = re.sub(r"‘[^‘’]*>\s*사이’", "‘" + MK["sa"] + " > 사이’", q39)
q39 = re.sub(r"‘[^‘’]*>\s*턱’", "‘" + MK["teuk"] + " > 턱’", q39)
q39 = re.sub(r"‘[^‘’]*>\s*거의’", "‘거싀 > 거의’", q39)
q39 = re.sub(r"[\ue000-\uf8ff\u200b]", "", q39)
for it in items:
    if it["id"] == "kor-eonmae-39":
        it["context"] = q39
json.dump(items, open("data/korean.json", "w"), ensure_ascii=False, indent=1)
print(len(items), "items; points total:", {s: sum(i["points"] for i in items if i["section"] in ("common", s)) for s in ("hwajak", "eonmae")})
for it in items:
    print(it["id"], it["answer"], it["points"], len(it["context"]))

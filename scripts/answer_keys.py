"""Parse KICE answer-key PDFs (홀수형, page 1) into {section: {number: (answer, points)}}."""
import re, subprocess
CIR = "①②③④⑤"
TOK = re.compile(r"(\d+)\s+([①-⑤]|\d+)\s+(\d)(?=\s|$)")


def _val(a):
    return CIR.index(a) + 1 if a in CIR else int(a)


def _rows(pdf):
    txt = subprocess.run(["pdftotext", "-layout", "-f", "1", "-l", "1", pdf, "-"], capture_output=True, text=True).stdout
    for line in txt.splitlines():
        toks = TOK.findall(line)
        if toks:
            yield [(int(q), _val(a), int(p)) for q, a, p in toks]


def math_key(pdf="data/pdf/27DaeSuneung_2a.pdf"):
    key = {"common": {}, "prob": {}, "calc": {}, "geom": {}}
    for toks in _rows(pdf):
        common = [t for t in toks[:2] if t[0] <= 22]
        for q, a, p in common:
            key["common"][q] = (a, p)
        rest = toks[len(common):]
        if len(rest) == 3:
            for sec, (q, a, p) in zip(("prob", "calc", "geom"), rest):
                key[sec][q] = (a, p)
    assert len(key["common"]) == 22 and all(len(key[s]) == 8 for s in ("prob", "calc", "geom")), {k: len(v) for k, v in key.items()}
    return key


def english_key(pdf="data/pdf/27DaeSuneung_3a.pdf"):
    key = {}
    for toks in _rows(pdf):
        for q, a, p in toks:
            key[q] = (a, p)
    assert len(key) == 45, len(key)
    return key


if __name__ == "__main__":
    k = math_key()
    for s, v in k.items():
        print(s, sum(p for _, p in v.values()), v)

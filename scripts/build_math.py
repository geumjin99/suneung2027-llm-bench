"""Build math items from the VLM page transcription (data/text/math_transcription.json) + answer key.
Manual corrections after checking against page images go in data/text/math_fixes.json ({item_id: {field: value}})."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from answer_keys import math_key

raw = json.load(open("data/text/math_transcription_raw.json"))
fixes = json.load(open("data/text/math_fixes.json")) if os.path.exists("data/text/math_fixes.json") else {}
key = math_key()
SEC_PAGES = {"common": range(1, 9), "prob": range(9, 13), "calc": range(13, 17), "geom": range(17, 21)}
items, problems = [], []
for sec, pages in SEC_PAGES.items():
    found = {}
    for p in pages:
        txt = raw[f"math_p{p:02d}.png"].strip()
        txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
        try:
            arr = json.loads(txt)
        except json.JSONDecodeError as e:
            problems.append(f"page {p}: JSON error {e}"); continue
        for q in arr:
            found[int(q["number"])] = q
    for n, (ans, pts) in sorted(key[sec].items()):
        iid = f"math-{sec}-{n:02d}"
        q = found.get(n)
        if q is None:
            problems.append(f"{iid}: missing in transcription"); continue
        it = {"id": iid, "subject": "math", "section": sec, "number": n,
              "type": "mcq" if n <= 15 or 23 <= n <= 28 else "short", "points": pts, "answer": ans,
              "question": q.get("question", ""), "figure": q.get("figure", ""), "choices": q.get("choices", [])}
        it["choices"] = [re.sub(r"^\s*[①-⑤]\s*", "", c) for c in it["choices"]]
        it.update(fixes.get(iid, {}))
        if it["type"] == "mcq" and len(it["choices"]) != 5:
            problems.append(f"{iid}: {len(it['choices'])} choices")
        if q.get("points") and int(q["points"]) != pts:
            problems.append(f"{iid}: points {q['points']} vs key {pts}")
        items.append(it)
json.dump(items, open("data/math.json", "w"), ensure_ascii=False, indent=1)
print(len(items), "items"); print("\n".join(problems) or "no structural problems")

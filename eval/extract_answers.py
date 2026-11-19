"""Lenient answer extraction ("washing") with an LLM reader.

For every sample in out/raw/*.json, an extractor model reads the final part of the response
(after the reasoning block) and reports which answer(s) the response finally commits to.
It does not judge correctness. Results are written next to the raw file as *.lenient.json:
    {item_id: [[answers for sample 0], [answers for sample 1], ...]}

Usage: python extract_answers.py <raw_dir> [--model Qwen/Qwen3.8-27B-FP8]
"""
import argparse, glob, json, os, re
from vllm import LLM, SamplingParams

PROMPT = """아래는 어떤 응시자가 시험 문항에 대해 작성한 답안의 마지막 부분입니다.
이 응시자가 최종적으로 고른 답을 찾아 주세요. 정답이 무엇인지 스스로 풀거나 판단하지 말고, 응시자가 쓴 내용만 근거로 하세요.

문항 유형: {kind}
규칙:
- 응시자가 최종 답으로 제시한 값만 적습니다. 풀이 중간에 검토만 하고 버린 후보는 제외합니다.
- 최종 답을 여러 개 제시했다면 모두 적습니다.
- 최종 답이 전혀 없으면 빈 리스트를 적습니다.
- {rule}

JSON만 출력하세요: {{"answers": [...]}}

[답안의 마지막 부분]
{tail}"""

MCQ_RULE = "5지선다형이므로 답은 1~5의 정수로 적습니다(①=1, ②=2, ③=3, ④=4, ⑤=5)."
SHORT_RULE = "단답형이므로 답은 0 이상 999 이하의 정수로 적습니다."


def tail_of(text, n=2500):
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text[-n:]


def parse(out, typ):
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return []
    try:
        vals = json.loads(m.group(0)).get("answers", [])
    except json.JSONDecodeError:
        return []
    res = []
    for v in vals:
        t = str(v).strip()
        if t and t[0] in "①②③④⑤":
            v = "①②③④⑤".index(t[0]) + 1
        else:
            m2 = re.match(r"-?\d+", t)
            if not m2:
                continue
            v = int(m2.group(0))
        if typ == "mcq" and not 1 <= v <= 5:
            continue
        if typ == "short" and not 0 <= v <= 999:
            continue
        if v not in res:
            res.append(v)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    a = ap.parse_args()
    files = [f for f in sorted(glob.glob(os.path.join(a.raw_dir, "*.json")))
             if not f.endswith(".lenient.json") and not os.path.exists(f[:-5] + ".lenient.json")]
    if not files:
        print("nothing to do"); return
    llm = LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    for f in files:
        d = json.load(open(f))
        typ_of = {}
        msgs, keys = [], []
        for r in d["records"]:
            typ = "short" if (r["subject"] == "math" and not (r["number"] <= 15 or 23 <= r["number"] <= 28)) else "mcq"
            typ_of[r["id"]] = typ
            for k, s in enumerate(r["samples"]):
                kind = "5지선다형" if typ == "mcq" else "단답형(정수)"
                rule = MCQ_RULE if typ == "mcq" else SHORT_RULE
                msgs.append([{"role": "user", "content": PROMPT.format(kind=kind, rule=rule, tail=tail_of(s["text"]))}])
                keys.append((r["id"], k))
        outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
        res = {}
        for (iid, k), o in zip(keys, outs):
            res.setdefault(iid, []).append(parse(o.outputs[0].text, typ_of[iid]))
        json.dump(res, open(f[:-5] + ".lenient.json", "w"))
        print("DONE", os.path.basename(f), flush=True)


if __name__ == "__main__":
    main()

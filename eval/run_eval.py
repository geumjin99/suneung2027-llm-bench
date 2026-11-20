"""Evaluate one model on the 2027 CSAT items with vLLM (offline).

Usage: python run_eval.py <hf_model_id> <items.json> <out_dir> [--n 3] [--max-tokens 16384]
"""
import argparse, json, os, re, time
from vllm import LLM, SamplingParams

SYSTEM = "당신은 대한민국 대학수학능력시험을 치르는 수험생입니다. 문제를 꼼꼼히 풀고 정답을 고르세요."
CIRCLED = "①②③④⑤"


def build_prompt(it):
    if it["subject"] == "math":
        q = f"{it['number']}. {it['question']} [{it['points']}점]"
        if it.get("figure"):
            q += f"\n\n[그림 설명] {it['figure']}"
        if it["type"] == "mcq":
            q += "\n\n" + "\n".join(f"{CIRCLED[i]} {c}" for i, c in enumerate(it["choices"]))
            tail = "풀이 과정을 쓴 뒤, 마지막 줄에 반드시 '정답: X' 형식으로 답을 적으세요. X는 ①, ②, ③, ④, ⑤ 중 하나입니다."
        else:
            tail = "풀이 과정을 쓴 뒤, 마지막 줄에 반드시 '정답: N' 형식으로 답을 적으세요. N은 0 이상 999 이하의 정수 하나입니다."
        head = f"다음은 수학 영역 문항입니다.\n\n{q}\n\n"
        return head + tail
    # korean / english: shared context block containing the passage and the question
    subj = it.get("area") or {"korean": "국어", "english": "영어"}[it["subject"]]
    return (f"다음은 {subj} 영역 문제지의 일부입니다.\n\n{it['context']}\n\n"
            f"위 자료를 바탕으로 {it['number']}번 문항에 답하세요. "
            "풀이 과정을 간단히 쓴 뒤, 마지막 줄에 반드시 '정답: X' 형식으로 답을 적으세요. X는 ①, ②, ③, ④, ⑤ 중 하나입니다.")


def strip_think(text):
    # remove reasoning blocks if the template leaves them in the content
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text


def extract(text, typ, finish=None):
    # a truncated generation that never closed its reasoning block has no final answer
    if finish == "length" and "</think>" not in text:
        return None
    t = strip_think(text)
    m = list(re.finditer(r"정답\s*[:：은는]?\s*\**\s*([①-⑤]|\d+)", t))
    cand = m[-1].group(1) if m else None
    if cand is None:
        # fallbacks: last circled digit, or \boxed{}
        b = re.findall(r"\\boxed\{\s*([^{}]+?)\s*\}", t)
        if b:
            cand = b[-1]
        else:
            c = re.findall(r"[①-⑤]", t[-300:])
            cand = c[-1] if c else None
    if cand is None:
        return None
    cand = cand.strip()
    if cand in CIRCLED:
        return CIRCLED.index(cand) + 1
    d = re.match(r"-?\d+", cand)
    if not d:
        return None
    v = int(d.group(0))
    if typ == "mcq" and not 1 <= v <= 5:
        return None
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("items", help="comma-separated item files"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=32768)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = a.model.replace("/", "__")
    todo = [f for f in a.items.split(",") if not os.path.exists(f"{a.out}/{tag}__{os.path.basename(f)}")]
    if not todo:
        print("nothing to do"); return
    t0 = time.time()
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
    tc = getattr(cfg, "text_config", None) or cfg
    native = getattr(tc, "max_position_embeddings", None) or a.max_model_len
    a.max_model_len = int(min(a.max_model_len, native))
    print("max_model_len", a.max_model_len, flush=True)
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=0.88,
              trust_remote_code=True,
              limit_mm_per_prompt={"image": 0, "audio": 0} if ("gemma-4" in a.model or "Qwen3.5" in a.model) else None)
    load_s = time.time() - t0
    base = llm.get_default_sampling_params()  # honours the model's generation_config.json
    sp = SamplingParams(n=a.n, max_tokens=a.max_tokens, seed=0,
                        temperature=base.temperature if base.temperature is not None else 0.7,
                        top_p=base.top_p if base.top_p is not None else 0.95,
                        top_k=base.top_k if base.top_k is not None else -1)
    for f in todo:
        run_one(llm, sp, a, f, tag, load_s)


def run_one(llm, sp, a, path, tag, load_s):
    items = json.load(open(path))
    msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(it)}] for it in items]
    t1 = time.time()
    tok = llm.get_tokenizer()

    def per_item_sp(ms):
        out = []
        for m in ms:
            try:
                n = len(tok.apply_chat_template(m, tokenize=True, add_generation_prompt=True, enable_thinking=True))
            except Exception:
                n = len(tok.encode("\n".join(x["content"] for x in m)))
            cap = max(256, min(a.max_tokens, a.max_model_len - n - 32))
            q = sp.clone(); q.max_tokens = cap; out.append(q)
        return out
    try:
        outs = llm.chat(msgs, per_item_sp(msgs), chat_template_kwargs={"enable_thinking": True})
    except Exception as e:  # templates without system role support
        print("retry without system role:", repr(e)[:300], flush=True)
        msgs = [[{"role": "user", "content": SYSTEM + "\n\n" + build_prompt(it)}] for it in items]
        outs = llm.chat(msgs, per_item_sp(msgs), chat_template_kwargs={"enable_thinking": True})
    gen_s = time.time() - t1
    recs, ntok = [], 0
    for it, o in zip(items, outs):
        samples = []
        for c in o.outputs:
            pred = extract(c.text, it["type"], c.finish_reason)
            ntok += len(c.token_ids)
            samples.append({"pred": pred, "correct": pred == it["answer"], "tokens": len(c.token_ids),
                            "finish": c.finish_reason, "text": c.text})
        recs.append({"id": it["id"], "subject": it["subject"], "section": it["section"], "number": it["number"],
                     "points": it["points"], "answer": it["answer"], "samples": samples})
    meta = {"model": a.model, "n": a.n, "max_tokens": a.max_tokens, "temperature": sp.temperature, "top_p": sp.top_p,
            "top_k": sp.top_k, "load_s": round(load_s, 1), "gen_s": round(gen_s, 1), "gen_tokens": ntok,
            "items": os.path.basename(path)}
    out = f"{a.out}/{tag}__{os.path.basename(path)}"
    json.dump({"meta": meta, "records": recs}, open(out + ".tmp", "w"), ensure_ascii=False)
    os.replace(out + ".tmp", out)
    acc = sum(s["correct"] for r in recs for s in r["samples"]) / (len(recs) * a.n)
    print(f"DONE {a.model} {path} acc={acc:.3f} gen_s={gen_s:.0f} tokens={ntok}", flush=True)


if __name__ == "__main__":
    main()

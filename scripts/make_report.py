"""Aggregate per-model result files into scores, figures and README.md.

Reads results/raw/*.json (full outputs, git-ignored) and writes:
  results/scores.csv, results/per_question.csv, figures/*.png, README.md
"""
import csv, glob, json, os, re, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# model metadata: display name, params (B, total), developer
MODELS = {
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B": ("HyperCLOVAX-SEED 0.5B", 0.57, "네이버"),
    "Qwen/Qwen3.5-0.8B": ("Qwen3.5-0.8B", 0.8, "Alibaba"),
    "LGAI-EXAONE/EXAONE-4.0-1.2B": ("EXAONE-4.0-1.2B", 1.28, "LG AI연구원"),
    "Qwen/Qwen3.5-2B": ("Qwen3.5-2B", 2.0, "Alibaba"),
    "google/gemma-4-E2B-it": ("Gemma-4-E2B", 2.0, "Google"),
    "kakaocorp/kanana-2-3b-instruct": ("kanana-2-3B", 3.0, "카카오"),
    "Qwen/Qwen3.5-4B": ("Qwen3.5-4B", 4.0, "Alibaba"),
    "google/gemma-4-E4B-it": ("Gemma-4-E4B", 4.0, "Google"),
    "skt/A.X-4.0-Light": ("A.X-4.0-Light", 7.3, "SKT"),
    "kakaocorp/kanana-1.5-8b-instruct-2505": ("kanana-1.5-8B", 8.0, "카카오"),
    "Qwen/Qwen3.5-9B": ("Qwen3.5-9B", 9.0, "Alibaba"),
    "K-intelligence/Midm-2.0-Base-Instruct": ("Mi:dm-2.0-Base", 11.5, "KT"),
    "google/gemma-4-12B-it": ("Gemma-4-12B", 12.0, "Google"),
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B": ("HyperCLOVAX-SEED-Think-14B", 14.7, "네이버"),
    "Qwen/Qwen3-14B": ("Qwen3-14B", 14.8, "Alibaba"),
}
# Approximate memory for 4-bit quantized weights (GB) and the device class that can hold it.
# Gemma-4 E2B/E4B figures follow Google's on-device numbers (per-layer embeddings kept off the accelerator).
MEM_4BIT = {
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B": 0.4, "Qwen/Qwen3.5-0.8B": 0.6,
    "LGAI-EXAONE/EXAONE-4.0-1.2B": 0.8, "Qwen/Qwen3.5-2B": 1.4, "google/gemma-4-E2B-it": 2.0,
    "kakaocorp/kanana-2-3b-instruct": 2.0, "Qwen/Qwen3.5-4B": 2.6, "google/gemma-4-E4B-it": 3.0,
    "kakaocorp/kanana-1.5-8b-instruct-2505": 4.8, "Qwen/Qwen3.5-9B": 5.5, "google/gemma-4-12B-it": 7.5,
    "skt/A.X-4.0-Light": 4.4, "K-intelligence/Midm-2.0-Base-Instruct": 6.9,
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B": 8.8,
    "Qwen/Qwen3-14B": 8.8,
}
DEVICE_TIERS = [  # (max GB for weights, label)
    (1.5, "보급형 스마트폰"), (3.2, "중급 스마트폰"), (6.0, "플래그십 스마트폰"), (99, "노트북·태블릿"),
]


def device_tier(hf_id):
    gb = MEM_4BIT.get(hf_id)
    if gb is None:
        return "–"
    return next(label for lim, label in DEVICE_TIERS if gb <= lim)


COLORS = {"Alibaba": "#4C72B0", "Google": "#55A868", "LG AI연구원": "#C44E52", "카카오": "#DD8452", "네이버": "#8172B3"}

# subject -> list of (score name, sections included)
EXAMS = {
    "korean": [("국어(화작)", ("common", "hwajak")), ("국어(언매)", ("common", "eonmae"))],
    "math": [("수학(확통)", ("common", "prob")), ("수학(미적)", ("common", "calc")), ("수학(기하)", ("common", "geom"))],
    "english": [("영어", ("all",))],
}


def _plain_ticks(ax):
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8, 14]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def load():
    res = defaultdict(dict)  # model -> subject -> data
    for f in glob.glob(os.path.join(RAW, "*.json")):
        if f.endswith(".lenient.json"):
            continue
        d = json.load(open(f))
        len_path = f[:-5] + ".lenient.json"
        lenient = json.load(open(len_path)) if os.path.exists(len_path) else None
        d["meta"]["washed"] = lenient is not None
        for r in d["records"]:
            for k, s in enumerate(r["samples"]):
                s["strict_correct"] = bool(s["correct"])
                s["strict_pred"] = s["pred"]
                truncated = s["finish"] == "length" and "</think>" not in s["text"]
                if truncated:
                    answers = []
                elif lenient is not None:
                    answers = lenient.get(r["id"], [[]] * len(r["samples"]))[k]
                else:
                    answers = [s["pred"]] if s["pred"] is not None else []
                s["answers"] = answers
                is_mcq = isinstance(r["answer"], int) and not (r["subject"] == "math" and r["id"].split("-")[1] in ("common", "prob", "calc", "geom") and (16 <= r["number"] <= 22 or r["number"] >= 29))
                if answers:
                    s["correct"] = (r["answer"] in answers) / len(answers)
                else:
                    s["correct"] = 0.2 if is_mcq else 0.0
                # pred used for majority vote / agreement: single committed answer or None
                s["pred"] = answers[0] if len(answers) == 1 else None
        subj = d["records"][0]["subject"]
        res[d["meta"]["model"]][subj] = d
    return res


def exam_scores(d, sections):
    """Expected raw score (mean over samples) and per-sample scores."""
    recs = [r for r in d["records"] if r["section"] in sections]
    n = len(recs[0]["samples"])
    per_sample = [sum(r["points"] * r["samples"][k]["correct"] for r in recs) for k in range(n)]
    # majority vote score
    maj = 0
    for r in recs:
        preds = [s["pred"] for s in r["samples"] if s["pred"] is not None]
        if preds:
            top = max(set(preds), key=preds.count)
            maj += r["points"] if top == r["answer"] else 0
    full = sum(r["points"] for r in recs)
    return statistics.mean(per_sample), (round(min(per_sample)), round(max(per_sample))), maj, full


def main():
    global RES
    res = load()
    RES = res
    rows = []
    for m, subjects in res.items():
        name, params, dev = MODELS.get(m, (m.split("/")[-1], None, "?"))
        row = {"model": name, "hf_id": m, "params_B": params, "developer": dev}
        for subj, d in subjects.items():
            for label, secs in EXAMS[subj]:
                if subj == "english":
                    secs = tuple({r["section"] for r in d["records"]})
                mean, (lo, hi), maj, full = exam_scores(d, secs)
                row[label] = round(mean, 1)
                row[label + "_range"] = f"{lo}–{hi}"
                row[label + "_maj"] = maj
            recs = d["records"]
            samples = [s for r in recs for s in r["samples"]]
            row[f"{subj}_acc"] = round(sum(s["correct"] for s in samples) / len(samples), 3)
            row[f"{subj}_noanswer"] = round(sum(s["strict_pred"] is None for s in samples) / len(samples), 3)
            row[f"{subj}_strict_acc"] = round(sum(s["strict_correct"] for s in samples) / len(samples), 3)
            row[f"{subj}_trunc"] = round(sum(s["finish"] == "length" for s in samples) / len(samples), 3)
            row[f"{subj}_avg_tokens"] = round(statistics.mean(s["tokens"] for s in samples))
            row[f"{subj}_gen_s"] = d["meta"]["gen_s"]
        rows.append(row)
    rows.sort(key=lambda r: (r["params_B"] or 0))

    labels = [l for subj in EXAMS for l, _ in EXAMS[subj]]
    present = [l for l in labels if any(l in r for r in rows)]

    # CSV
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    keys = ["model", "hf_id", "params_B", "developer"] + sum(([l, l + "_range", l + "_maj"] for l in present), []) + \
        sorted({k for r in rows for k in r if k.endswith(("_acc", "_noanswer", "_strict_acc", "_trunc", "_avg_tokens", "_gen_s"))})
    with open(os.path.join(ROOT, "results", "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow({k: r.get(k, "") for k in keys}) for r in rows]

    # per-question correctness (no question text): model, item id, answer, preds
    with open(os.path.join(ROOT, "results", "per_question.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "item", "points", "answer", "preds", "n_correct", "avg_tokens"])
        for m, subjects in res.items():
            for d in subjects.values():
                for r in d["records"]:
                    w.writerow([MODELS.get(m, (m,))[0], r["id"], r["points"], r["answer"],
                                "|".join(str(s["pred"]) for s in r["samples"]),
                                sum(s["correct"] for s in r["samples"]),
                                round(statistics.mean(s["tokens"] for s in r["samples"]))])

    # Figures
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    try:
        import matplotlib.font_manager as fm
        for cand in ["Noto Sans CJK KR", "Noto Sans CJK SC", "UnDotum"]:
            if any(cand in f.name for f in fm.fontManager.ttflist):
                plt.rcParams["font.family"] = cand; break
    except Exception:
        pass

    figs = []
    # 1) grouped bar of exam scores
    if rows and present:
        fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(rows) + 2), 4.6))
        width = 0.8 / len(present)
        pal = ["#4C72B0", "#8DA0CB", "#55A868", "#9BCB8F", "#C7E9C0", "#DD8452"]
        for j, l in enumerate(present):
            xs = [i + (j - (len(present) - 1) / 2) * width for i in range(len(rows))]
            ys = [r.get(l, 0) or 0 for r in rows]
            ax.bar(xs, ys, width, label=l, color=pal[j % len(pal)])
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["model"] for r in rows], rotation=30, ha="right")
        ax.set_ylabel("원점수 (100점 만점, 3회 평균)"); ax.set_ylim(0, 100)
        ax.set_title("2027학년도 수능 — 소형 모델 원점수")
        ax.legend(ncol=min(6, len(present)), fontsize=8, frameon=False, loc="upper left")
        fig.tight_layout(); p = os.path.join(FIG, "scores_by_model.png"); fig.savefig(p, dpi=160); plt.close(fig)
        figs.append(("영역별 원점수", "figures/scores_by_model.png"))

    # 2) score vs params scatter per subject (best elective)
    for subj, title in (("korean", "국어"), ("math", "수학"), ("english", "영어")):
        pts = []
        for r in rows:
            cand = [r[l] for l, _ in EXAMS[subj] if l in r]
            if cand and r["params_B"]:
                pts.append((r["params_B"], statistics.mean(cand), r["model"], r["developer"]))
        if len(pts) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for x, y, n, dv in pts:
            ax.scatter(x, y, s=60, color=COLORS.get(dv, "gray"), zorder=3)
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_xscale("log"); _plain_ticks(ax); ax.set_xlabel("매개변수 수 (B, 로그 축)"); ax.set_ylabel("원점수 (선택과목 평균)")
        ax.set_ylim(0, 100); ax.set_title(f"{title}: 모델 크기 vs 점수"); ax.grid(alpha=.3)
        handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in COLORS.items()]
        ax.legend(handles=handles, fontsize=8, frameon=False)
        fig.tight_layout(); p = os.path.join(FIG, f"scaling_{subj}.png"); fig.savefig(p, dpi=160); plt.close(fig)
        figs.append((f"{title}: 크기 vs 점수", f"figures/scaling_{subj}.png"))

    # 3) per-question heatmap (fraction correct over samples)
    for subj, title in (("korean", "국어"), ("math", "수학"), ("english", "영어")):
        ms = [r for r in rows if subj in res[r["hf_id"]]]
        if not ms:
            continue
        items = [x["id"] for x in res[ms[0]["hf_id"]][subj]["records"]]
        mat = []
        for r in ms:
            byid = {x["id"]: x for x in res[r["hf_id"]][subj]["records"]}
            mat.append([sum(s["correct"] for s in byid[i]["samples"]) / len(byid[i]["samples"]) if i in byid else float("nan") for i in items])
        fig, ax = plt.subplots(figsize=(max(8, 0.17 * len(items) + 2), 0.38 * len(ms) + 1.6))
        im = ax.imshow(mat, aspect="auto", cmap="Greens", vmin=0, vmax=1)
        ax.set_yticks(range(len(ms))); ax.set_yticklabels([r["model"] for r in ms], fontsize=8)
        short = [re.sub(r"^[a-z]+-", "", i).replace("common-", "").replace("hwajak-", "화").replace("eonmae-", "언")
                 .replace("prob-", "확").replace("calc-", "미").replace("geom-", "기").replace("all-", "") for i in items]
        ax.set_xticks(range(len(items))); ax.set_xticklabels(short, fontsize=6, rotation=90)
        ax.set_title(f"{title}: 문항별 정답률 (3회 중)"); fig.colorbar(im, ax=ax, fraction=0.02)
        fig.tight_layout(); p = os.path.join(FIG, f"heatmap_{subj}.png"); fig.savefig(p, dpi=160); plt.close(fig)
        figs.append((f"{title}: 문항별 정답률", f"figures/heatmap_{subj}.png"))

    hero(rows)
    import accuracy_figs
    acc_figs, tables = accuracy_figs.make(res, rows, FIG, COLORS) if rows else ([], {})
    write_readme(rows, present, figs, acc_figs, tables)


K_LABELS = ["국어(화작)", "국어(언매)"]
M_LABELS = ["수학(확통)", "수학(미적)", "수학(기하)"]

# UI strings per language
T = {
    "ko": {
        "tpl": "README.template.md", "out": "README.md",
        "cols": ["모델", "개발사", "매개변수(B)", "실행 가능 기기"],
        "label": lambda l: l,
        "diag": "{s} 정제 후 / 형식 엄수 / 형식 위반 / 잘림 / 평균 토큰", "subj": ("국어", "수학", "영어"),
        "unsolved": "**모든 모델이 3회 모두 틀린 문항**: ", "all_solved": "**모든 모델이 3회 모두 맞힌 문항**: ",
        "dev_hdr": ["기기 등급", "대략적인 기기 사양", "4비트 가중치 크기", "해당 모델", "이 등급 최고 점수 (국어 / 수학 / 영어)"],
        "tier": {"보급형 스마트폰": "보급형 스마트폰", "중급 스마트폰": "중급 스마트폰", "플래그십 스마트폰": "플래그십 스마트폰", "노트북·태블릿": "노트북·태블릿"},
        "spec": {"보급형 스마트폰": "RAM 4–6GB", "중급 스마트폰": "RAM 8GB", "플래그십 스마트폰": "RAM 12–16GB", "노트북·태블릿": "RAM 16GB 이상"},
        "rng": {"보급형 스마트폰": "≤1.5GB", "중급 스마트폰": "1.5–3.2GB", "플래그십 스마트폰": "3.2–6GB", "노트북·태블릿": "6GB 이상"},
        "dev": {"네이버": "네이버", "LG AI연구원": "LG AI연구원", "카카오": "카카오", "Alibaba": "알리바바", "Google": "구글"},
    },
}


def _best_in(r, labels):
    v = [r[l] for l in labels if l in r]
    return max(v) if v else None


def _avg_in(r, labels):
    v = [r[l] for l in labels if l in r]
    return statistics.mean(v) if v else None


def write_readme(rows, present, figs, acc_figs=(), tables=None):
    tables = tables or {}
    from datetime import datetime, timezone, timedelta
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
    for lang, L in T.items():
        tpl = open(os.path.join(ROOT, L["tpl"])).read()
        lines = ["| " + " | ".join(L["cols"] + [L["label"](l) for l in present]) + " |",
                 "|" + "---|" * (4 + len(present))]
        for r in rows:
            cells = [f"{r[l]:.1f} <sub>({r[l + '_range']})</sub>" if l in r else "—" for l in present]
            lines.append(f"| {r['model']} | {L['dev'].get(r['developer'], r['developer'])} | {r['params_B']} | "
                         f"{L['tier'].get(device_tier(r['hf_id']), '–')} | " + " | ".join(cells) + " |")
        diag = ["| " + L["cols"][0] + " | " + " | ".join(L["diag"].format(s=s) for s in L["subj"]) + " |", "|---|---|---|---|"]
        for r in rows:
            c = [f"{r[s + '_acc']:.2f} / {r[s + '_strict_acc']:.2f} / {r[s + '_noanswer']:.2f} / {r[s + '_trunc']:.2f} / {r[s + '_avg_tokens']}"
                 if s + "_acc" in r else "—" for s in ("korean", "math", "english")]
            diag.append(f"| {r['model']} | " + " | ".join(c) + " |")
        figmd = "\n\n".join(f"**{t}**\n\n![{t}]({p})" for t, p in figs)
        accmd = "\n\n".join(f"**{t}**\n\n![{t}]({p})" for t, p in acc_figs)
        sep = ", "
        notes = []
        if tables.get("unsolved"):
            notes.append(L["unsolved"] + sep.join(f"`{x}`" for x in sorted(tables["unsolved"])))
        if tables.get("all_solved"):
            notes.append(L["all_solved"] + sep.join(f"`{x}`" for x in sorted(tables["all_solved"])))
        accmd += ("\n\n" + "\n\n".join(notes)) if notes else ""
        dev = ["| " + " | ".join(L["dev_hdr"]) + " |", "|---|---|---|---|---|"]
        for _, label in DEVICE_TIERS:
            ms = [r for r in rows if device_tier(r["hf_id"]) == label]
            kv = [x for x in (_best_in(r, K_LABELS) for r in ms) if x is not None]
            mv = [x for x in (_best_in(r, M_LABELS) for r in ms) if x is not None]
            ev = [x for x in (_best_in(r, ["영어"]) for r in ms) if x is not None]
            dev.append(f"| {L['tier'][label]} | {L['spec'][label]} | {L['rng'][label]} | {', '.join(r['model'] for r in ms) or '–'} | "
                       f"{f'{max(kv):.0f}' if kv else '–'} / {f'{max(mv):.0f}' if mv else '–'} / {f'{max(ev):.0f}' if ev else '–'} |")
        out = (tpl.replace("{{ACC_FIGURES}}", accmd).replace("{{DEVICE_TABLE}}", "\n".join(dev))
               .replace("{{SCORES_TABLE}}", "\n".join(lines)).replace("{{DIAG_TABLE}}", "\n".join(diag))
               .replace("{{FIGURES}}", figmd).replace("{{UPDATED}}", stamp).replace("{{N_MODELS}}", str(len(rows)))
               .replace("{{SUMMARY}}", summary(rows)))
        open(os.path.join(ROOT, L["out"]), "w").write(out)



def hero(rows):
    """Headline chart: best score reachable on each device class."""
    import numpy as np
    tiers = [label for _, label in DEVICE_TIERS]
    subj = [("국어", K_LABELS, "#4C72B0"), ("수학", M_LABELS, "#DD8452"), ("영어", ["영어"], "#55A868")]
    subj = [x for x in subj if any(_best_in(r, x[1]) is not None for r in rows)]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.8 / max(1, len(subj))
    for j, (name, labels, col) in enumerate(subj):
        for i, t in enumerate(tiers):
            ms = [r for r in rows if device_tier(r["hf_id"]) == t and _best_in(r, labels) is not None]
            if not ms:
                continue
            v, r = max(((_best_in(r, labels), r) for r in ms), key=lambda x: x[0])
            x = i + (j - (len(subj) - 1) / 2) * w
            ax.bar(x, v, w * 0.92, color=col, label=name if i == 0 or not ax.get_legend_handles_labels()[1].count(name) else None)
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")
            ax.text(x, 3, r["model"].replace("HyperCLOVAX-SEED ", "HCX "), ha="center", va="bottom", rotation=90, fontsize=7, color="white")
    h, l = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    ax.legend(seen.values(), seen.keys(), frameon=False, ncol=3, loc="upper left")
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"{t}\n({ {'보급형 스마트폰': 'RAM 4–6GB', '중급 스마트폰': 'RAM 8GB', '플래그십 스마트폰': 'RAM 12–16GB', '노트북·태블릿': 'RAM 16GB+'}[t]})" for t in tiers])
    ax.set_ylim(0, 105); ax.set_ylabel("최고 원점수 (100점 만점)")
    ax.set_title("인터넷 없이 기기에서 돌아가는 AI의 2027 수능 최고 점수", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "hero.png"), dpi=170); plt.close(fig)


RES = {}


def summary(rows):
    """Plain-language bullet points generated from the current results."""
    out = []

    def best(labels):
        cand = [(_best_in(r, labels), r) for r in rows if _best_in(r, labels) is not None]
        return max(cand, key=lambda t: t[0]) if cand else None
    k, m, e = best(K_LABELS), best(M_LABELS), best(["영어"])
    tops = []
    for v, name_ko in ((k, "국어"), (m, "수학"), (e, "영어")):
        if v:
            tops.append(f"{name_ko} **{v[0]:.0f}점**({v[1]['model']})")
    if tops:
        out.append("- **가장 높은 점수**: " + ", ".join(tops) + ". (100점 만점 원점수, 3회 평균)")
    tiers = []
    for _, label in DEVICE_TIERS:
        ms = [r for r in rows if device_tier(r["hf_id"]) == label]
        kv = [x for x in (_best_in(r, K_LABELS) for r in ms) if x is not None]
        mv = [x for x in (_best_in(r, M_LABELS) for r in ms) if x is not None]
        if not (kv or mv):
            continue
        ev = [x for x in (_best_in(r, ["영어"]) for r in ms) if x is not None]
        parts = []
        if kv: parts.append(f"국어 최고 {max(kv):.0f}점")
        if mv: parts.append(f"수학 최고 {max(mv):.0f}점")
        if ev: parts.append(f"영어 최고 {max(ev):.0f}점")
        tiers.append(f"{label}급 모델은 " + "·".join(parts))
    if tiers:
        out.append("- **기기별로 보면** " + ", ".join(tiers) + "이다. (아래 표 참고, 메모리 기준 추정)")
    e90 = sorted([r for r in rows if (_best_in(r, ["영어"]) or 0) >= 90], key=lambda r: r["params_B"] or 0)
    if e90:
        r0 = e90[0]
        out.append(f"- **영어**는 절대평가라 90점 이상이면 1등급이다. {device_tier(r0['hf_id'])}에서 돌아가는 {r0['params_B']:g}B 모델"
                   f"({r0['model']}, {r0['영어']:.0f}점)부터 1등급 점수가 나왔고, 1등급 점수를 받은 모델은 모두 {len(e90)}개다. "
                   "단, 듣기는 음성 대신 대본을 읽고 풀었다.")
    small = [r for r in rows if (r["params_B"] or 0) <= 1.0 and "korean_acc" in r]
    if small:
        acc = statistics.mean(statistics.mean(r[f"{s}_acc"] for s in ("korean", "math", "english") if f"{s}_acc" in r) for r in small)
        out.append(f"- **1B 이하 모델**은 평균 정답률이 {acc * 100:.0f}%로, 5지선다를 무작위로 찍었을 때(20%)보다 조금 나은 수준이다. "
                   "생각을 끝내지 못하고 길이 제한에 걸리거나, 답을 정해진 형식으로 쓰지 못하는 경우가 많았다.")
    reach = sorted([r for r in rows if (_best_in(r, M_LABELS) or 0) >= 80], key=lambda r: r["params_B"] or 0)
    if reach:
        mk = reach[0]
        out.append(f"- **수학**은 작은 모델도 잘 푼다. {mk['params_B']:g}B 크기의 {mk['model']}가 80점을 넘었다.")
    kill = []
    for r in rows:
        d = RES.get(r["hf_id"], {}).get("math")
        if d:
            ks = [x for x in d["records"] if x["points"] == 4 and x["number"] in (21, 22, 29, 30)]
            kill.append(statistics.mean(s["correct"] for x in ks for s in x["samples"]))
    if kill:
        out.append(f"- 시험지 맨 끝의 4점 단답형 고난도 문항(공통 21·22번, 선택 29·30번)은 모델 평균 정답률이 {statistics.mean(kill) * 100:.0f}%, "
                   f"가장 잘 푼 모델도 {max(kill) * 100:.0f}%에 그쳤다.")
    both = [r for r in rows if "korean_acc" in r and "math_acc" in r and (r["params_B"] or 0) >= 2]
    if both:
        gap = statistics.mean(r["math_acc"] - r["korean_acc"] for r in both)
        if gap > 0.03:
            out.append(f"- **국어가 수학보다 어렵다.** 2B 이상 모델 기준으로 국어 정답률이 수학보다 평균 {gap * 100:.0f}%p 낮다. "
                       "2,000–4,000자 길이의 지문을 읽고 선택지 사이의 미묘한 차이를 가려내는 일이 작은 모델에게는 계산보다 어렵다.")
    kr = [r for r in rows if r["developer"] in ("네이버", "LG AI연구원", "카카오")]
    if kr:
        def f(v):
            return f"{v:.0f}" if v is not None else "–"
        def desc(r):
            parts = [f"{n} {_avg_in(r, ls):.0f}점" for n, ls in (("국어", K_LABELS), ("수학", M_LABELS), ("영어", ["영어"])) if _avg_in(r, ls) is not None]
            return f"{r['model']}(" + ", ".join(parts) + ")"
        out.append("- **국산 모델**: " + ", ".join(desc(r) for r in kr) + ".")
    return "\n".join(out) if out else "- 평가가 진행 중이다."


if __name__ == "__main__":
    main()

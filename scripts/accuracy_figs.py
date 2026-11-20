"""Accuracy-centred figures. Called from make_report.py with the loaded results.

Every figure is about sample-level accuracy (fraction of the 3 samples x items answered correctly),
so models are comparable regardless of point weighting.
"""
import os, statistics
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# question-type buckets: (subject, label, predicate on record)
def _plain_ticks(ax):
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8, 14]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _korean_bucket(r):
    if r["section"] == "common":
        return "국어·독서(1–17)" if r["number"] <= 17 else "국어·문학(18–34)"
    return {"hwajak": "국어·화법과 작문", "eonmae": "국어·언어와 매체"}[r["section"]]


def _math_bucket(r):
    if r["section"] == "common":
        return "수학·공통 5지선다" if r["number"] <= 15 else "수학·공통 단답형"
    return {"prob": "수학·확률과 통계", "calc": "수학·미적분", "geom": "수학·기하"}[r["section"]]


def _english_bucket(r):
    return "영어·듣기(1–17)" if r["number"] <= 17 else "영어·읽기(18–45)"


SOCIAL = {"ethics": "생활과 윤리", "thought": "윤리와 사상", "economics": "경제", "politics": "정치와 법", "society": "사회·문화"}
BUCKET = {"korean": _korean_bucket, "math": _math_bucket, "english": _english_bucket,
          "history": lambda r: "한국사", "social": lambda r: "사탐·" + SOCIAL[r["section"]]}
BUCKET_ORDER = ["국어·독서(1–17)", "국어·문학(18–34)", "국어·화법과 작문", "국어·언어와 매체",
                "수학·공통 5지선다", "수학·공통 단답형", "수학·확률과 통계", "수학·미적분", "수학·기하",
                "영어·듣기(1–17)", "영어·읽기(18–45)", "한국사"] + ["사탐·" + v for v in SOCIAL.values()]
SUBJ_CN = {"korean": "국어", "math": "수학", "english": "영어", "history": "한국사", "social": "사회탐구"}


def _acc(recs):
    s = [x["correct"] for r in recs for x in r["samples"]]
    return sum(s) / len(s) if s else float("nan")


def _save(fig, fig_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, name), dpi=160)
    plt.close(fig)
    return f"figures/{name}"


def make(res, rows, fig_dir, colors):
    """res: model -> subject -> {meta, records}; rows: sorted model rows (with 'hf_id','model','params_B','developer')."""
    figs, tables = [], {}
    names = [r["model"] for r in rows]

    # ---------- 1. accuracy heat table: model x question type ----------
    cols = [b for b in BUCKET_ORDER if any(
        BUCKET[s](rec) == b for r in rows for s, d in res[r["hf_id"]].items() for rec in d["records"])]
    if cols:
        mat = np.full((len(rows), len(cols) + 1), np.nan)
        for i, r in enumerate(rows):
            allrec = []
            for s, d in res[r["hf_id"]].items():
                allrec += d["records"]
                for j, b in enumerate(cols):
                    recs = [x for x in d["records"] if BUCKET[s](x) == b]
                    if recs:
                        mat[i, j] = _acc(recs)
            mat[i, -1] = _acc(allrec)
        labels = cols + ["전체"]
        fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 3, 0.42 * len(rows) + 1.8))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j] * 100:.0f}", ha="center", va="center", fontsize=8,
                            color="black" if 0.25 < mat[i, j] < 0.8 else "white")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(names, fontsize=8)
        ax.axvline(len(cols) - 0.5, color="white", lw=3)
        ax.set_title("정답률(%): 모델 × 문항 유형   (5지선다 무작위 = 20%)")
        fig.colorbar(im, ax=ax, fraction=0.025)
        figs.append(("정답률 총괄표: 모델 × 문항 유형", _save(fig, fig_dir, "acc_heatmap.png")))
        tables["acc_by_type"] = (labels, names, mat)

    # ---------- 2. accuracy vs params, one line per subject ----------
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for s, mk in (("korean", "o"), ("math", "s"), ("english", "^"), ("history", "D"), ("social", "v")):
        pts = [(r["params_B"], _acc(res[r["hf_id"]][s]["records"]), r) for r in rows
               if s in res[r["hf_id"]] and r["params_B"]]
        if not pts:
            continue
        pts.sort(key=lambda t: t[0])
        for x, y, r in pts:
            ax.scatter(x, y * 100, marker=mk, s=55, color=colors.get(r["developer"], "gray"), zorder=3,
                       edgecolor="black", linewidth=0.4)
        ax.scatter([], [], marker=mk, color="white", edgecolor="black", label=SUBJ_CN[s])
    ax.axhline(20, color="red", ls="--", lw=0.8); ax.text(0.45, 21.5, "무작위 20% (5지선다)", color="red", fontsize=7)
    ax.set_xscale("log"); _plain_ticks(ax); ax.set_xlabel("매개변수 수 (B, 로그 축)"); ax.set_ylabel("정답률 (%)"); ax.set_ylim(0, 100)
    ax.grid(alpha=.3); ax.set_title("모델 크기에 따른 정답률")
    h1 = ax.legend(loc="upper left", fontsize=8, frameon=False, title="모양 = 과목", title_fontsize=8)
    ax.add_artist(h1)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in colors.items()],
              loc="lower right", fontsize=8, frameon=False, title="색 = 개발사", title_fontsize=8)
    figs.append(("모델 크기 vs 정답률", _save(fig, fig_dir, "acc_vs_params.png")))

    # ---------- 3. accuracy by point value (difficulty proxy) ----------
    subs = [s for s in ("korean", "math", "english", "history", "social") if any(s in res[r["hf_id"]] for r in rows)]
    if subs:
        fig, axes = plt.subplots(1, len(subs), figsize=(4.2 * len(subs), 3.8), sharey=True, squeeze=False)
        cmap = plt.get_cmap("viridis")
        for ax, s in zip(axes[0], subs):
            pv = sorted({x["points"] for r in rows if s in res[r["hf_id"]] for x in res[r["hf_id"]][s]["records"]})
            for i, r in enumerate(rows):
                if s not in res[r["hf_id"]]:
                    continue
                recs = res[r["hf_id"]][s]["records"]
                ys = [_acc([x for x in recs if x["points"] == p]) * 100 for p in pv]
                ax.plot(pv, ys, marker="o", color=cmap(i / max(1, len(rows) - 1)), label=r["model"], lw=1.2)
            ax.set_xticks(pv); ax.set_xticklabels([f"{p}점" for p in pv]); ax.set_title(SUBJ_CN[s])
            ax.set_ylim(0, 100); ax.grid(alpha=.3)
        axes[0][0].set_ylabel("정답률 (%)")
        axes[0][-1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.suptitle("배점별 정답률 (배점이 높을수록 대체로 어려움)", fontsize=11)
        figs.append(("난이도별: 2·3·4점 문항 정답률", _save(fig, fig_dir, "acc_by_points.png")))

    # ---------- 4. failure decomposition (stacked) ----------
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(rows) + 2), 4.2))
    cats = [("정답", "#4C9A5B"), ("오답", "#E0A458"), ("형식 위반(답 없음)", "#B0B0B0"), ("길이 초과로 잘림", "#C44E52")]
    vals = {c: [] for c, _ in cats}
    for r in rows:
        ss = [x for d in res[r["hf_id"]].values() for rec in d["records"] for x in rec["samples"]]
        n = len(ss)
        trunc = sum(x["finish"] == "length" for x in ss)
        noans = sum(x["strict_pred"] is None and x["finish"] != "length" for x in ss)
        corr = sum(x["strict_correct"] for x in ss)
        vals["정답"].append(corr / n * 100); vals["길이 초과로 잘림"].append(trunc / n * 100)
        vals["형식 위반(답 없음)"].append(noans / n * 100); vals["오답"].append((n - corr - trunc - noans) / n * 100)
    bottom = np.zeros(len(rows))
    for c, col in cats:
        ax.bar(range(len(rows)), vals[c], bottom=bottom, color=col, label=c); bottom += np.array(vals[c])
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("전체 풀이 중 비율 (%)"); ax.set_ylim(0, 100); ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center",
                                                                   bbox_to_anchor=(0.5, 1.12))
    ax.set_title("답안 결과 구성 (전 과목, 형식 엄수 기준)", pad=26)
    figs.append(("오답 유형 구성", _save(fig, fig_dir, "failure_breakdown.png")))

    # ---------- 5. consistency: mean vs majority vote, agreement ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    mean_acc, maj_acc, agree = [], [], []
    for r in rows:
        recs = [x for d in res[r["hf_id"]].values() for x in d["records"]]
        mean_acc.append(_acc(recs) * 100)
        mj, ag = 0, 0
        for x in recs:
            preds = [s["pred"] for s in x["samples"]]
            valid = [p for p in preds if p is not None]
            if valid:
                top = max(set(valid), key=valid.count)
                mj += top == x["answer"]
            ag += len(set(preds)) == 1
        maj_acc.append(mj / len(recs) * 100); agree.append(ag / len(recs) * 100)
    xs = np.arange(len(rows))
    a1.bar(xs - 0.2, mean_acc, 0.4, label="1회 평균", color="#8DA0CB")
    a1.bar(xs + 0.2, maj_acc, 0.4, label="3회 다수결", color="#4C72B0")
    a1.set_xticks(xs); a1.set_xticklabels(names, rotation=35, ha="right", fontsize=8); a1.set_ylim(0, 100)
    a1.set_ylabel("정답률 (%)"); a1.legend(fontsize=8, frameon=False); a1.set_title("다수결로 얼마나 오르나")
    a2.bar(xs, agree, color="#55A868"); a2.set_xticks(xs); a2.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    a2.set_ylim(0, 100); a2.set_ylabel("3회 답이 모두 같은 문항 비율 (%)"); a2.set_title("답의 일관성")
    figs.append(("일관성: 다수결과 답의 안정성", _save(fig, fig_dir, "consistency.png")))

    # ---------- 6. reasoning length vs accuracy ----------
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for r in rows:
        ss = [x for d in res[r["hf_id"]].values() for rec in d["records"] for x in rec["samples"]]
        t = statistics.mean(x["tokens"] for x in ss); a = sum(x["correct"] for x in ss) / len(ss) * 100
        ax.scatter(t, a, s=60, color=colors.get(r["developer"], "gray"), edgecolor="black", linewidth=0.4, zorder=3)
        ax.annotate(r["model"], (t, a), textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xscale("log"); _plain_ticks(ax); ax.set_xlabel("평균 출력 길이 (토큰, 로그 축)"); ax.set_ylabel("정답률 (%)")
    ax.set_ylim(0, 100); ax.grid(alpha=.3); ax.set_title("오래 생각할수록 더 정확할까?")
    figs.append(("추론 길이 vs 정답률", _save(fig, fig_dir, "tokens_vs_acc.png")))

    # ---------- 7. hardest / easiest items ----------
    per_item = defaultdict(list)
    for r in rows:
        for s, d in res[r["hf_id"]].items():
            for x in d["records"]:
                per_item[x["id"]].append(sum(y["correct"] for y in x["samples"]) / len(x["samples"]))
    full = {k: statistics.mean(v) for k, v in per_item.items() if len(v) == len(rows)}
    if full:
        hard = sorted(full.items(), key=lambda kv: kv[1])[:15]
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.barh(range(len(hard))[::-1], [v * 100 for _, v in hard], color="#C44E52")
        ax.set_yticks(range(len(hard))[::-1]); ax.set_yticklabels([k for k, _ in hard], fontsize=8)
        ax.set_xlabel("전체 모델 평균 정답률 (%)"); ax.set_xlim(0, 100); ax.set_title("가장 어려운 15문항")
        figs.append(("가장 어려운 15문항", _save(fig, fig_dir, "hardest_items.png")))
        tables["unsolved"] = [k for k, v in full.items() if v == 0]
        tables["all_solved"] = [k for k, v in full.items() if v == 1]
    return figs, tables

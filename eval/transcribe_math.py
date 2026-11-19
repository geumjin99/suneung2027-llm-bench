"""Transcribe KICE math exam page images to structured JSON with a VLM (vLLM offline)."""
import json, glob, base64, os, sys
from vllm import LLM, SamplingParams

MODEL = os.environ.get("TRANSCRIBE_MODEL", "Qwen/Qwen3.8-27B-FP8")
PROMPT = """이 이미지는 한국 대학수학능력시험 수학 영역 문제지의 한 페이지입니다.
페이지에 있는 모든 문항을 빠짐없이, 원문 그대로 전사하세요.

규칙:
- 수식은 모두 LaTeX로 정확히 옮기고 $...$ 로 감쌉니다. 기호·첨자·지수·극한 방향(예: x\\to a-)을 정확히 옮깁니다.
- 한국어 문장은 원문 그대로 옮기고 요약하거나 고치지 않습니다.
- (가), (나) 같은 조건 상자, <보기>의 ㄱ,ㄴ,ㄷ 항목도 그대로 옮깁니다.
- 그림·그래프가 있으면 "figure" 필드에 문제를 푸는 데 필요한 정보(점의 좌표, 곡선의 모양, 교점, 표시된 길이·각도, 색칠된 영역 등)를 한국어로 자세하고 객관적으로 서술합니다. 그림이 없으면 빈 문자열.
- 5지선다형이면 "choices"에 ①~⑤ 다섯 개 선택지를 순서대로 넣고, 단답형이면 빈 리스트.
- "points"는 [2점]/[3점]/[4점] 표기의 숫자.

JSON 배열만 출력하세요(코드블록 없이):
[{"number": 문항번호(정수), "points": 정수, "question": "문항 본문", "figure": "그림 설명", "choices": ["①의 내용", ...]}]"""

def main():
    pages = sorted(glob.glob("img/math_p*.png"))
    llm = LLM(model=MODEL, max_model_len=32768, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=12000)
    msgs = []
    for p in pages:
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        msgs.append([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": PROMPT}]}])
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    os.makedirs("../out", exist_ok=True)
    res = {os.path.basename(p): o.outputs[0].text for p, o in zip(pages, outs)}
    json.dump(res, open("../out/math_transcription_raw.json", "w"), ensure_ascii=False, indent=1)
    print("DONE", len(res))

if __name__ == "__main__":
    main()

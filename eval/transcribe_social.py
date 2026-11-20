"""Transcribe KICE Korean history / social studies pages to Markdown with a VLM (vLLM offline).

Usage: python transcribe_social.py history ethics thought economics politics society
Reads img/<subject>_pNN.png, writes ../out/social_transcription_raw.json ({filename: text}).
"""
import json, glob, base64, os, sys
from vllm import LLM, SamplingParams
MODEL = os.environ.get("TRANSCRIBE_MODEL", "Qwen/Qwen3.8-27B-FP8")
PROMPT = """이 이미지는 한국 대학수학능력시험 4교시(한국사 또는 사회탐구) 문제지의 한 페이지입니다(2단 편집).
페이지 내용을 읽는 순서(왼쪽 단 위→아래, 그다음 오른쪽 단 위→아래)대로 빠짐없이, 원문 그대로 전사하세요.

규칙:
- 각 문항은 "1. 질문 [3점]" 처럼 문항 번호로 시작하는 줄로 적습니다. 글자를 고치거나 요약하지 않습니다.
- (가) (나) ㉠ ⓐ A B 갑 을 <보기> ㄱ. ㄴ. ①~⑤ 같은 기호를 원문 위치에 그대로 둡니다. 네모 칸 안의 기호(예: 네모 속 (가))는 (가)로 적습니다. 밑줄 친 부분은 <u>...</u>로 표시합니다.
- 표는 Markdown 표로 옮기고, 칸의 수치와 기호(○, ×, 빈칸 등)를 정확히 적습니다.
- 그래프는 [그래프: ...] 형식으로 축 이름, 단위, 각 항목의 수치나 대소 관계를 빠짐없이 적습니다.
- 말풍선, 대화, 카드, 게시판, 칠판, 신문 기사 같은 이미지 속 글자는 모두 전사하고, 발화자를 "발화자: 내용" 형식으로 적습니다. 한자는 원문 그대로 적습니다.
- 사진·그림은 [그림: ...] 형식으로 보이는 모습만 객관적으로 서술합니다. 이미지 안에 글자로 적혀 있지 않은 유물·건물·인물·장소의 이름은 추측해서 쓰지 않습니다.
- 지도는 [지도: ...] 형식으로 표시된 지역, 기호, 화살표, 글자를 서술합니다.
- 쪽 번호, 머리말(홀수형, 영역 이름, 교시, 성명·수험 번호 칸, 선택 과목 표시), 저작권 문구, '확인 사항' 상자는 생략합니다.
전사 결과만 출력하세요."""


def main():
    subs = sys.argv[1:]
    pages = sorted(p for s in subs for p in glob.glob(f"img/{s}_p*.png"))
    llm = LLM(model=MODEL, max_model_len=32768, limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=12000)
    msgs = [[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()}},
        {"type": "text", "text": PROMPT}]}] for p in pages]
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    os.makedirs("../out", exist_ok=True)
    json.dump({os.path.basename(p): o.outputs[0].text for p, o in zip(pages, outs)},
              open("../out/social_transcription_raw.json", "w"), ensure_ascii=False, indent=1)
    print("DONE", len(pages))


if __name__ == "__main__":
    main()

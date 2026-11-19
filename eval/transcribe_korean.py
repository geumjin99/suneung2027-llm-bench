"""Transcribe KICE Korean exam pages to Markdown with a VLM (vLLM offline)."""
import json, glob, base64, os
from vllm import LLM, SamplingParams
MODEL = os.environ.get("TRANSCRIBE_MODEL", "Qwen/Qwen3.8-27B-FP8")
PROMPT = """이 이미지는 한국 대학수학능력시험 국어 영역 문제지의 한 페이지입니다(2단 편집).
페이지 내용을 읽는 순서(왼쪽 단 위→아래, 그다음 오른쪽 단 위→아래)대로 빠짐없이, 원문 그대로 전사하세요.

규칙:
- 글자를 고치거나 요약하지 않습니다. 줄바꿈은 문단 단위로만 둡니다.
- ㉠ ⓐ [A] (가) <보기> ①~⑤ 같은 기호를 원문 위치에 그대로 둡니다. 밑줄이나 괄호로 묶인 구간 표시([A] 등)는 해당 구간 앞에 "[A]" 처럼 표시합니다.
- 표는 Markdown 표로 옮깁니다.
- 대화방 화면, 슬라이드, 포스터 같은 이미지 속 글자도 모두 전사하고, 발화자 이름을 "이름: 내용" 형식으로 적습니다.
- 그림·사진처럼 글자가 아닌 시각 정보는 [그림: ...] 형식으로 문제 풀이에 필요한 내용을 객관적으로 서술합니다.
- 옛한글(15세기 국어)은 가능한 한 유니코드 옛한글 자모로 정확히 옮깁니다.
- 쪽 번호, 머리말(홀수형, 국어 영역), 저작권 문구, '확인 사항' 상자는 생략합니다.
전사 결과만 출력하세요."""
def main():
    pages = sorted(glob.glob("img/korean_p*.png"))
    llm = LLM(model=MODEL, max_model_len=32768, limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=12000)
    msgs = [[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()}},
        {"type": "text", "text": PROMPT}]}] for p in pages]
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    os.makedirs("../out", exist_ok=True)
    json.dump({os.path.basename(p): o.outputs[0].text for p, o in zip(pages, outs)},
              open("../out/korean_transcription_raw.json", "w"), ensure_ascii=False, indent=1)
    print("DONE", len(pages))
if __name__ == "__main__":
    main()

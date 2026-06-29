"""ChatGPT를 이용한 어휘 의미 분별(WSD).

문장(문맥)과 각 단어의 우리말샘 후보 센스(어깨번호/정의/원어/예문)를 함께 주고,
문맥에 가장 적절한 센스 하나를 고르게 한다. 단어 전체를 한 번의 호출로 처리한다.
"""
import json
from typing import Dict, List

from openai import AsyncOpenAI

from .config import get_settings
from .schemas import Morpheme, ResolvedSense, SenseCandidate

_SYSTEM_PROMPT = """\
너는 한국어 어휘 의미 분별(WSD) 전문가다. 우리말샘 사전을 사용한다.
입력으로 문장과, 그 문장에 등장한 단어별 후보 의미(센스) 목록을 받는다.
각 단어에 대해 '문맥상' 가장 알맞은 센스를 정확히 하나 고른다.

판단 근거: 정의(definition), 원어(origin), 예문(examples)을 모두 고려한다.
적절한 후보가 전혀 없으면 target_code 를 null 로 둔다.
confidence 는 high/medium/low 중 하나로 스스로 평가한다.

반드시 아래 JSON 형식 그대로만 출력한다. 설명 금지. 입력의 index를 그대로 돌려준다.
{"results": [{"index": 0, "target_code": 123, "sup_no": "2", "sense_no": "1",
              "definition": "선택한 정의 원문", "origin": "원어 또는 null",
              "confidence": "high|medium|low"}]}
"""


def _build_payload(morphemes: List[Morpheme], senses: Dict[str, List[SenseCandidate]]) -> dict:
    words = []
    for idx, m in enumerate(morphemes):
        cands = senses.get(m.lemma, [])
        words.append(
            {
                "index": idx,
                "surface": m.surface,
                "lemma": m.lemma,
                "pos": m.pos,
                "candidates": [
                    {
                        "target_code": c.target_code,
                        "sup_no": c.sup_no,
                        "sense_no": c.sense_no,
                        "pos": c.pos,
                        "definition": c.definition,
                        "origin": c.origin,
                        "examples": c.examples[:3],
                    }
                    for c in cands
                ],
            }
        )
    return {"words": words}


async def disambiguate(
    sentence: str,
    morphemes: List[Morpheme],
    senses: Dict[str, List[SenseCandidate]],
) -> Dict[int, ResolvedSense]:
    # 후보가 있는 단어만 WSD 대상으로 보낸다.
    targets = [m for m in morphemes if senses.get(m.lemma)]
    if not targets:
        return {}

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    payload = _build_payload(morphemes, senses)
    user_msg = (
        f"문장: {sentence}\n\n"
        f"단어 및 후보 의미(JSON):\n{json.dumps(payload, ensure_ascii=False)}"
    )

    resp = await client.chat.completions.create(
        model=settings.openai_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    data = json.loads(resp.choices[0].message.content or "{}")

    resolved: Dict[int, ResolvedSense] = {}
    for r in data.get("results", []):
        idx = r.get("index")
        if idx is None or r.get("target_code") is None:
            continue
        resolved[int(idx)] = ResolvedSense(
            target_code=r.get("target_code"),
            sup_no=r.get("sup_no"),
            sense_no=r.get("sense_no"),
            definition=r.get("definition"),
            origin=r.get("origin"),
            confidence=r.get("confidence", "low"),
        )
    return resolved

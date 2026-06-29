"""ChatGPT를 이용한 한국어 형태소 분석.

문장에서 명사/동사/부사/형용사만 추출하고 각 단어의 원형(lemma)을 복원한다.
"""
import json
from typing import List

from openai import AsyncOpenAI

from .config import get_settings
from .schemas import Morpheme

_ALLOWED_POS = {"명사", "동사", "부사", "형용사"}

_SYSTEM_PROMPT = """\
너는 정밀한 한국어 형태소 분석기다. 입력 문장을 분석해 내용어만 추출한다.

규칙:
- 추출 대상 품사: 명사, 동사, 부사, 형용사 (이 네 가지 외에는 모두 제외)
- 조사/어미/접사/관형사/감탄사/대명사/수사 등은 제외한다.
- lemma(원형) 규칙:
  - 동사·형용사: 사전 표제어 형태인 '-다'로 끝나는 기본형으로 복원한다. (먹었다→먹다, 예뻤다→예쁘다, 갔다→가다)
  - 명사·부사: 표면형에서 조사를 제거한 기본형으로 쓴다. (배를→배, 빨리→빨리)
- 복합어/파생어는 우리말샘 표제어로 존재할 법한 단위로 끊는다.
- 같은 단어가 여러 번 나오면 등장한 만큼 모두 출력한다(표면형 순서 유지).

반드시 아래 JSON 형식 그대로만 출력한다. 설명 금지.
{"tokens": [{"surface": "표면형", "lemma": "원형", "pos": "명사|동사|부사|형용사"}]}
"""


async def analyze_morphemes(sentence: str) -> List[Morpheme]:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    resp = await client.chat.completions.create(
        model=settings.openai_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": sentence},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)

    morphemes: List[Morpheme] = []
    for tok in data.get("tokens", []):
        pos = (tok.get("pos") or "").strip()
        lemma = (tok.get("lemma") or "").strip()
        surface = (tok.get("surface") or lemma).strip()
        if pos in _ALLOWED_POS and lemma:
            morphemes.append(Morpheme(surface=surface, lemma=lemma, pos=pos))
    return morphemes

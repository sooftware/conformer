"""한국어 형태소분석 + 우리말샘 의미(어깨번호) 결정 API.

흐름:
  문장 → (GPT) 형태소분석 → (우리말샘) 후보 센스 → (GPT) WSD → JSON/XML
"""
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response

from .morph import analyze_morphemes
from .urimalsaem import lookup_senses
from .schemas import AnalyzeRequest, AnalyzeResponse, AnalyzedToken
from .wsd import disambiguate
from .xml_utils import to_xml

app = FastAPI(
    title="Korean Morph + 우리말샘 Sense API",
    description="문장을 형태소분석해 명사·동사·부사·형용사의 원형과 우리말샘 어깨번호·정의를 반환합니다.",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _run_pipeline(sentence: str) -> AnalyzeResponse:
    # 1) 형태소 분석 (명사/동사/부사/형용사 + 원형)
    morphemes = await analyze_morphemes(sentence)

    # 2) 우리말샘에서 원형별 후보 센스 수집
    senses = await lookup_senses([m.lemma for m in morphemes])

    # 3) 문맥 기반 WSD로 어깨번호/정의 확정
    resolved = await disambiguate(sentence, morphemes, senses)

    # 4) 조립
    tokens = []
    for idx, m in enumerate(morphemes):
        tokens.append(
            AnalyzedToken(
                surface=m.surface,
                lemma=m.lemma,
                pos=m.pos,
                sense=resolved.get(idx),
                candidates_count=len(senses.get(m.lemma, [])),
            )
        )
    return AnalyzeResponse(sentence=sentence, tokens=tokens)


@app.post("/analyze")
async def analyze(
    req: AnalyzeRequest,
    format: str = Query("json", pattern="^(json|xml)$", description="응답 형식"),
):
    result = await _run_pipeline(req.sentence)
    if format == "xml":
        return Response(content=to_xml(result), media_type="application/xml")
    return JSONResponse(content=result.model_dump())


# GET 편의용 (브라우저에서 바로 테스트)
@app.get("/analyze")
async def analyze_get(
    sentence: str = Query(..., min_length=1),
    format: str = Query("json", pattern="^(json|xml)$"),
):
    result = await _run_pipeline(sentence)
    if format == "xml":
        return Response(content=to_xml(result), media_type="application/xml")
    return JSONResponse(content=result.model_dump())

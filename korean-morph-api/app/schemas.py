from typing import List, Optional

from pydantic import BaseModel, Field


# ---- 형태소 분석 결과 ----
class Morpheme(BaseModel):
    surface: str = Field(..., description="문장에 나타난 표면형 (예: 배가)")
    lemma: str = Field(..., description="원형 (동사/형용사는 '-다' 형, 예: 먹다)")
    pos: str = Field(..., description="명사 | 동사 | 부사 | 형용사")


# ---- 우리말샘 후보 센스 ----
class SenseCandidate(BaseModel):
    target_code: Optional[int] = None
    word: str
    sup_no: Optional[str] = Field(None, description="어깨번호 (동형어 구분 번호)")
    sense_no: Optional[str] = None
    pos: Optional[str] = None
    definition: Optional[str] = None
    origin: Optional[str] = Field(None, description="원어 (한자/외래어 등)")
    examples: List[str] = Field(default_factory=list)


# ---- WSD로 확정된 센스 ----
class ResolvedSense(BaseModel):
    target_code: Optional[int] = None
    sup_no: Optional[str] = None
    sense_no: Optional[str] = None
    definition: Optional[str] = None
    origin: Optional[str] = None
    confidence: str = Field("low", description="high | medium | low")


# ---- 최종 토큰 ----
class AnalyzedToken(BaseModel):
    surface: str
    lemma: str
    pos: str
    sense: Optional[ResolvedSense] = None
    candidates_count: int = 0


# ---- 요청/응답 ----
class AnalyzeRequest(BaseModel):
    sentence: str = Field(..., min_length=1, examples=["나는 배를 타고 바다로 나갔다."])


class AnalyzeResponse(BaseModel):
    sentence: str
    tokens: List[AnalyzedToken]

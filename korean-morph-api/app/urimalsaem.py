"""우리말샘(개방형 한국어 지식 대사전) Open API 클라이언트.

- search API: 표제어로 후보 센스(어깨번호=sup_no, 정의) 목록 조회
- view   API: target_code로 상세(원어/예문) 조회

응답은 XML(req_type=xml)로 받아 ElementTree로 파싱한다.
참고: https://opendict.korean.go.kr/service/openApiInfo
"""
import asyncio
from typing import List, Optional
from xml.etree import ElementTree as ET

import httpx

from .config import get_settings
from .schemas import SenseCandidate


def _text(node: Optional[ET.Element]) -> Optional[str]:
    if node is None or node.text is None:
        return None
    return node.text.strip()


async def _search_one(client: httpx.AsyncClient, lemma: str) -> List[SenseCandidate]:
    settings = get_settings()
    params = {
        "key": settings.krdict_api_key,
        "q": lemma,
        "req_type": "xml",
        "part": "word",     # 표제어 검색
        "sort": "dict",     # 사전순
        "method": "exact",  # 정확히 일치하는 표제어만
        "num": 100,
    }
    r = await client.get(f"{settings.krdict_base_url}/search", params=params)
    r.raise_for_status()
    root = ET.fromstring(r.text)

    candidates: List[SenseCandidate] = []
    for item in root.iter("item"):
        target_code = _text(item.find("target_code"))
        word = _text(item.find("word")) or lemma
        sup_no = _text(item.find("sup_no"))
        # 한 item 안에 sense가 여러 개일 수 있다.
        for sense in item.iter("sense"):
            candidates.append(
                SenseCandidate(
                    target_code=int(target_code) if target_code and target_code.isdigit() else None,
                    word=word,
                    sup_no=sup_no,
                    sense_no=_text(sense.find("sense_no")),
                    pos=_text(sense.find("pos")),
                    definition=_text(sense.find("definition")),
                )
            )
    return candidates[: settings.max_senses_per_word]


async def _fill_detail(client: httpx.AsyncClient, cand: SenseCandidate) -> None:
    """view API로 원어(origin)와 예문(examples)을 채운다."""
    if cand.target_code is None:
        return
    settings = get_settings()
    params = {
        "key": settings.krdict_api_key,
        "method": "target_code",
        "q": cand.target_code,
        "req_type": "xml",
    }
    try:
        r = await client.get(f"{settings.krdict_base_url}/view", params=params)
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except (httpx.HTTPError, ET.ParseError):
        return

    item = root.find(".//item")
    if item is None:
        return

    word_info = item.find("word_info")
    if word_info is not None:
        cand.origin = _text(word_info.find("original_language")) or cand.origin

    # 해당 sense_no의 예문만 추출
    for sense_info in item.iter("sense_info"):
        if _text(sense_info.find("sense_no")) != cand.sense_no:
            continue
        for ex in sense_info.iter("example_info"):
            ex_text = _text(ex.find("example"))
            if ex_text:
                cand.examples.append(ex_text)


async def lookup_senses(lemmas: List[str]) -> dict[str, List[SenseCandidate]]:
    """여러 원형을 동시에 조회해 {lemma: [후보 센스...]} 로 반환한다."""
    settings = get_settings()
    unique = list(dict.fromkeys(lemmas))  # 중복 제거, 순서 유지

    limits = httpx.Limits(max_connections=10)
    async with httpx.AsyncClient(timeout=15.0, limits=limits) as client:
        results = await asyncio.gather(
            *[_search_one(client, lemma) for lemma in unique],
            return_exceptions=True,
        )
        table: dict[str, List[SenseCandidate]] = {}
        for lemma, res in zip(unique, results):
            table[lemma] = [] if isinstance(res, BaseException) else res

        if settings.fetch_sense_detail:
            detail_tasks = [
                _fill_detail(client, cand)
                for cands in table.values()
                for cand in cands
            ]
            await asyncio.gather(*detail_tasks, return_exceptions=True)

    return table

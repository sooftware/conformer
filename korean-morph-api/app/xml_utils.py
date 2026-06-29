"""AnalyzeResponse를 XML 문자열로 직렬화."""
from xml.etree import ElementTree as ET
from xml.dom import minidom

from .schemas import AnalyzeResponse


def _sub(parent: ET.Element, tag: str, text) -> ET.Element:
    el = ET.SubElement(parent, tag)
    if text is not None:
        el.text = str(text)
    return el


def to_xml(resp: AnalyzeResponse) -> str:
    root = ET.Element("analysis")
    _sub(root, "sentence", resp.sentence)
    tokens_el = ET.SubElement(root, "tokens")
    for t in resp.tokens:
        tok = ET.SubElement(tokens_el, "token")
        _sub(tok, "surface", t.surface)
        _sub(tok, "lemma", t.lemma)
        _sub(tok, "pos", t.pos)
        _sub(tok, "candidates_count", t.candidates_count)
        if t.sense is not None:
            s = ET.SubElement(tok, "sense")
            _sub(s, "target_code", t.sense.target_code)
            _sub(s, "sup_no", t.sense.sup_no)
            _sub(s, "sense_no", t.sense.sense_no)
            _sub(s, "definition", t.sense.definition)
            _sub(s, "origin", t.sense.origin)
            _sub(s, "confidence", t.sense.confidence)
    rough = ET.tostring(root, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

# 한국어 형태소분석 + 우리말샘 의미(어깨번호) API

임의의 한국어 문장을 입력하면 **명사·동사·부사·형용사**만 추려서
각 단어의 **원형**, **우리말샘 어깨번호(sup_no)**, **문맥에 맞는 센스 정의**를
JSON 또는 XML로 반환합니다.

## 처리 파이프라인

```
문장
 └─(1) ChatGPT      형태소분석 → 명사/동사/부사/형용사 + 원형(lemma)
 └─(2) 우리말샘 API  원형별 후보 센스 수집(어깨번호·정의·원어·예문)  [XML]
 └─(3) ChatGPT      문맥 + 후보 표제항으로 WSD → 최적 어깨번호·정의 1개
 └─(4) JSON / XML 반환
```

## 준비물 (인증키 2개)

1. **OpenAI API Key** — <https://platform.openai.com/api-keys>
2. **우리말샘 Open API 인증키** — <https://opendict.korean.go.kr/> → 오픈 API → 인증키 신청
   - (표준국어대사전 API <https://stdict.korean.go.kr/openapi/> 도 같은 코드로 사용 가능. `KRDICT_BASE_URL`만 교체)

## 로컬 실행

```bash
cd korean-morph-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # 키 입력
uvicorn app.main:app --reload --port 8000
```

브라우저에서 문서: <http://localhost:8000/docs>

## 호출 예시

```bash
# JSON
curl -X POST localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"sentence":"나는 배를 타고 바다로 나갔다."}'

# XML
curl -X POST 'localhost:8000/analyze?format=xml' \
  -H 'Content-Type: application/json' \
  -d '{"sentence":"맛있는 배를 먹었다."}'

# GET (브라우저 테스트용)
curl 'localhost:8000/analyze?sentence=빨리 달리는 말'
```

### 응답 예 (JSON)

```json
{
  "sentence": "나는 배를 타고 바다로 나갔다.",
  "tokens": [
    {
      "surface": "배를", "lemma": "배", "pos": "명사",
      "candidates_count": 8,
      "sense": {
        "target_code": 452178,
        "sup_no": "2",
        "sense_no": "1",
        "definition": "사람이나 짐 따위를 싣고 물 위로 떠다니도록 나무나 쇠 따위로 만든 물건.",
        "origin": null,
        "confidence": "high"
      }
    },
    {
      "surface": "타고", "lemma": "타다", "pos": "동사",
      "candidates_count": 6,
      "sense": { "sup_no": "1", "sense_no": "1", "definition": "탈것이나 짐승의 등 따위에 몸을 얹다.", "confidence": "high" }
    }
  ]
}
```

## AWS EC2 배포 (CPU 인스턴스)

```bash
# 1) 인스턴스: Ubuntu 22.04, t3.small 이상. 보안그룹 인바운드 80/443 허용.
# 2) 코드 업로드 후
sudo apt update && sudo apt install -y python3-venv nginx
cd ~/korean-morph-api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env          # 키 입력

# 3) systemd 서비스 등록 (앱은 127.0.0.1:8000에서 구동)
sudo cp deploy/morph-api.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now morph-api

# 4) Nginx 리버스 프록시 (80 → 8000)
sudo cp deploy/nginx.conf /etc/nginx/sites-available/morph-api
sudo ln -s /etc/nginx/sites-available/morph-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 5) (선택) HTTPS
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

도커로 띄우려면:

```bash
docker build -t morph-api .
docker run -d -p 8000:8000 --env-file .env --name morph-api morph-api
```

## 주의 / 튜닝 포인트

- **비용·지연**: 요청 1건당 GPT 호출 2회(형태소+WSD) + 우리말샘 호출 N회가 발생합니다.
  `FETCH_SENSE_DETAIL=false`로 예문 조회를 끄면 호출 수가 크게 줍니다.
- **우리말샘 호출량 제한**: 개방형 API는 일일 쿼터가 있습니다. 자주 쓰는 표제어는
  로컬 캐시(SQLite/Redis)로 보관하면 비용·지연을 모두 줄일 수 있습니다.
- **정확도**: 형태소 단위가 우리말샘 표제어와 어긋나면 후보가 0건일 수 있습니다.
  이때 `sense`는 `null`이고 `candidates_count: 0`으로 표시됩니다.
- LLM 제공자는 교체 가능합니다(`morph.py`/`wsd.py`만 수정). 현재는 요청하신 ChatGPT 사용.

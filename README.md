# 똑똑한 학습 전략성적키트

> Smart Learning Strategy & Grade Toolkit

학생 성적 분포를 예측하는 머신러닝 기반 API 시스템

## 프로젝트 개요

소수의 학생 점수 샘플로부터 전체 학급의 성적 분포를 예측하는 FastAPI 기반 백엔드 시스템입니다. SetTransformer 아키텍처를 활용한 딥러닝 모델로 히스토그램 예측을 수행합니다.

## 주요 기능

- **성적 히스토그램 예측**: 샘플 점수로부터 전체 학급의 성적 분포 예측
- **누적 히스토그램 생성**: 과목별 평가 항목들의 가중치를 고려한 누적 성적 분포
- **과목 및 평가 항목 관리**: 과목, 과제, 시험 등의 평가 항목 CRUD
- **학생 프로필 관리**: 학생 선호도 및 프로필 정보 관리
- **OpenAI API 통합**: AI 기반 학습 조언 제공 (과목별 학습 전략, 학기 계획)
- **Redis 캐시**: OpenAI API 응답 캐싱으로 빠른 응답 속도 제공

## 기술 스택

### Backend

- **FastAPI** 0.121.3+ - 고성능 웹 프레임워크
- **SQLAlchemy** 2.0.0+ - ORM 및 데이터베이스 관리
- **Pydantic** 2.12.4+ - 데이터 검증 및 스키마 정의
- **Uvicorn** 0.38.0+ - ASGI 서버
- **Redis** 5.0.0+ - 캐시 서버 (OpenAI API 응답 캐싱)

### Machine Learning

- **PyTorch** 2.9.1+ - 딥러닝 프레임워크
- **SetTransformer** - 집합 기반 예측 모델 아키텍처
- **NumPy** 2.3.5+ - 수치 연산
- **Weights & Biases** - 실험 추적 및 모델 관리

### Development Tools

- **Jupyter** - 모델 개발 및 실험
- **Matplotlib** - 데이터 시각화
- **OpenAI API** - AI 기능 통합

## 프로젝트 구조

```
2025-realthon/
├── main.py                    # FastAPI 애플리케이션 메인
├── hackathon.db               # SQLite 데이터베이스
├── pyproject.toml             # 프로젝트 의존성 관리
├── ML/                        # 머신러닝 모듈
│   ├── model_loader.py        # 모델 아키텍처 및 예측기
│   └── best_model_nnj359uw.pt # 학습된 모델 체크포인트
├── crawling/                  # 데이터 수집 모듈
└── init_db.py                 # 데이터베이스 초기화
```

## 설치 및 실행

### 요구사항

- Python 3.11
- uv (권장) 또는 pip

### 설치

```bash
# 저장소 클론
git clone https://github.com/BetaTester772/2025-realthon.git
cd 2025-realthon

# uv를 사용한 의존성 설치 (권장)
uv sync

# 또는 pip 사용
pip install -e .
```

### 환경 설정

`.env` 파일을 생성하고 필요한 환경 변수를 설정합니다:

```bash
# OpenAI API 설정 (필수)
OPENAI_API_KEY=your_api_key_here

# Redis 캐시 설정 (선택, 기본값 사용 가능)
REDIS_HOST=localhost    # 기본값: localhost
REDIS_PORT=6379         # 기본값: 6379
REDIS_DB=0              # 기본값: 0
CACHE_TTL=3600          # 캐시 유효 시간(초), 기본값: 3600 (1시간)
```

**Redis 설치 및 실행 (선택사항)**

Redis가 설치되어 있지 않으면 캐시가 비활성화되며, API는 정상 작동합니다.

```bash
# macOS (Homebrew)
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# Windows (WSL 사용 권장)
```

### 데이터베이스 초기화

```bash
python init_db.py
```

### 서버 실행

#### 방법 1: Docker Compose (권장)

```bash
# .env 파일에 OPENAI_API_KEY 설정 필요
# Docker Compose로 모든 서비스 시작 (FastAPI + Redis + Caddy)
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 서비스 중지
docker-compose down

# 볼륨까지 삭제 (Redis 데이터 포함)
docker-compose down -v
```

서비스 구성:
- **FastAPI App**: 백엔드 API 서버
- **Redis**: 캐시 서버 (OpenAI API 응답 캐싱)
- **Caddy**: 리버스 프록시 및 SSL 지원

API 문서는 `http://localhost/docs` (Caddy를 통한 접근) 또는 `http://localhost:8000/docs` (직접 접근)에서 확인할 수 있습니다.

#### 방법 2: 로컬 실행

```bash
# Redis 수동 실행 필요 (선택사항)
# macOS: brew services start redis
# Ubuntu: sudo systemctl start redis

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API 문서는 `http://localhost:8000/docs`에서 확인할 수 있습니다.

## API 엔드포인트

### ML Prediction

- `GET /predict-histogram` - 샘플 점수로부터 히스토그램 예측
    - Query Parameters: `evaluation_item_id`
    - 반환: 10개 구간의 성적 분포 + 사용자 점수/백분위 (my_score가 있는 경우)

- `GET /courses/{course_id}/cumulative-histogram` - 과목별 누적 히스토그램
    - 모든 평가 항목의 가중치를 고려한 누적 성적 분포
    - 반환: 가중 평균 누적 히스토그램 + 사용자 누적 점수/백분위 (my_score가 있는 경우)

### Course Management

- `GET /courses` - 전체 과목 목록 조회
- `POST /courses` - 새 과목 생성
- `GET /courses/{course_id}` - 특정 과목 조회
- `PUT /courses/{course_id}` - 과목 정보 수정
- `DELETE /courses/{course_id}` - 과목 삭제

### Evaluation Items

- `GET /evaluation-items` - 평가 항목 목록 조회
- `POST /evaluation-items` - 새 평가 항목 생성
- `GET /courses/{course_id}/evaluation-items` - 과목별 평가 항목 조회

### Student Profile

- `GET /student-profile` - 학생 프로필 조회
- `PUT /student-profile` - 학생 프로필 업데이트

### AI Advice (OpenAI API 통합)

- `GET /courses/{course_id}/advice` - 과목별 학습 조언
    - Query Parameters: `objective_grade` (목표 성적, 예: "A+", "B0")
    - 반환: 과제/시험 난이도 분석 + 목표 성적 달성 전략
    - Redis 캐시 사용 (동일한 course_id + objective_grade 조합에 대해 캐시)

- `GET /semester-advice` - 학기 전체 학습 계획
    - Query Parameters: `course_ids[]`, `target_grades[]`
    - 반환: 각 과목별 노력 배분 비율(%) + 전체 학기 조언
    - Redis 캐시 사용

### Cache Management

- `DELETE /cache/clear` - Redis 캐시 무효화
    - Query Parameters: `pattern` (기본값: "*", 모든 캐시)
    - 예시:
        - `?pattern=*` - 모든 캐시 삭제
        - `?pattern=course_advice:*` - 과목 조언 캐시만 삭제
        - `?pattern=semester_advice:*` - 학기 조언 캐시만 삭제

**주의**: 새로운 수강평이 추가되면 해당 과목의 모든 캐시가 자동으로 무효화됩니다.

자세한 API 명세는 `/docs` 엔드포인트에서 확인하세요.

## ML 모델 아키텍처

### FlexibleHistogramPredictor (SetTransformer 기반)

프로젝트는 **Set Transformer** 아키텍처를 기반으로 한 딥러닝 모델을 사용하여 소수의 샘플 점수로부터 전체 학급의 성적 분포를 예측합니다.

#### 모델 구조

```
Input (샘플 점수)
  → Input Projection (1 → hidden_dim)
  → SetTransformer Encoder (Inducing Points 기반)
  → Pooling (평균)
  → Decoder (hidden_dim → 10 bins)
  → Softmax
Output (히스토그램 확률 분포)
```

#### 핵심 컴포넌트

1. **MultiheadAttentionBlock**
    - Multi-head Self-Attention with residual connections
    - Layer Normalization
    - Feed-Forward Network (dim → 4×dim → dim)
    - Dropout regularization

2. **SetTransformerEncoder**
    - **Inducing Points (IP)**: 학습 가능한 latent representations (기본 16개)
    - 집합의 순서에 불변한(permutation-invariant) 인코딩
    - 두 단계 attention:
        1. IP가 입력 집합 attend → H 생성
        2. 입력 집합이 H attend → 인코딩된 표현

#### 하이퍼파라미터

| 파라미터           | 기본값 | 설명                |
|----------------|-----|-------------------|
| `hidden_dim`   | 64  | 임베딩 차원            |
| `num_heads`    | 4   | Attention head 수  |
| `num_inducers` | 16  | Inducing Points 수 |
| `dropout`      | 0.1 | Dropout 비율        |
| `num_bins`     | 10  | 히스토그램 구간 수        |

#### 입력/출력

- **입력**: 가변 길이 점수 샘플 (0-100 범위, 정규화 후 0-1)
- **출력**: 10개 구간의 확률 분포 (0-10, 10-20, ..., 90-100)
    - 확률값 (0-1) 또는 학생 수 (total_students 파라미터 지정 시)

#### 학습 데이터

합성 데이터 생성 기반 학습:

- **4가지 난이도 타입**:
    - `easy`: 평균 75-90점, 표준편차 5-10
    - `normal`: 평균 60-80점, 표준편차 8-15
    - `hard`: 평균 40-65점, 표준편차 8-15
    - `bimodal`: 이중 정규분포 (40-60점 그룹 + 70-90점 그룹)
- 각 클래스당 30명 학생 기준

### 모델 평가 지표

모델 성능은 다음 세 가지 지표로 측정됩니다:

- **MSE** (Mean Squared Error)
    - 예측 히스토그램과 실제 히스토그램의 평균 제곱 오차
    - 낮을수록 정확한 예측

- **JS Divergence** (Jensen-Shannon Divergence)
    - 두 확률 분포 간의 대칭적 거리 측정
    - 0에 가까울수록 두 분포가 유사

- **EMD** (Earth Mover's Distance = Wasserstein-1)
    - 한 분포를 다른 분포로 변환하는데 필요한 최소 "작업량"
    - 히스토그램의 형태적 유사성을 잘 포착

### 모델 사용

```python
from ML.model_loader import HistogramPredictor

# 모델 로드
predictor = HistogramPredictor("ML/best_model_nnj359uw.pt")

# 샘플 점수로 예측
sample_scores = [75, 82, 68, 91, 77, 85, 73, 80, 88, 79]
histogram = predictor.predict(sample_scores, total_students=99)

# 결과: {"0-10": 0, "10-20": 2, ..., "90-100": 5}
```

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

```
MIT License

Copyright (c) 2025 2025-realthon contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 문의

이슈가 있거나 질문이 있으시면 GitHub Issues를 이용해주세요.

## 관련 링크

- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [PyTorch 문서](https://pytorch.org/docs/)
- [SetTransformer 논문](https://arxiv.org/abs/1810.00825)
- [Weights & Biases](https://wandb.ai/)
"""
Smart Learning Strategy & Grade Toolkit API

FastAPI 기반 성적 분포 예측 및 학습 전략 추천 시스템.
SetTransformer 딥러닝 모델을 활용한 히스토그램 예측과 OpenAI API 기반 학습 조언을 제공합니다.
"""

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional
import os
from dotenv import load_dotenv
from openai import OpenAI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ OpenAI Client initialized successfully.")
else:
    print("⚠ Warning: OPENAI_API_KEY not found.")

DB_PATH = os.path.join(BASE_DIR, "hackathon.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class StudentProfileModel(Base):
    """학생 프로필 정보 저장 모델."""
    __tablename__ = "student_profile"
    id = Column(Integer, primary_key=True, index=True)
    preferences = Column(String)


class OtherStudentScoreModel(Base):
    """다른 학생들의 점수 데이터 저장 모델."""
    __tablename__ = "other_student_scores"
    id = Column(Integer, primary_key=True, index=True)
    evaluation_item_id = Column(Integer, index=True)
    score = Column(Float)


class CourseModel(Base):
    """과목 정보 저장 모델."""
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    course_code = Column(String, index=True)
    total_students = Column(Integer, default=99)


class EvaluationItemModel(Base):
    """평가 항목(과제, 시험 등) 정보 저장 모델."""
    __tablename__ = "evaluation_items"
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, index=True)
    name = Column(String)
    weight = Column(Integer)
    my_score = Column(Float, nullable=True)
    is_submitted = Column(Boolean, default=False)


class CourseReviewModel(Base):
    """과목 수강평 저장 모델."""
    __tablename__ = "course_reviews"
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, index=True)
    content = Column(String)


Base.metadata.create_all(bind=engine)


class StudentProfileCreate(BaseModel):
    """학생 프로필 생성 요청 스키마."""
    preferences: str


class StudentProfileResponse(BaseModel):
    """학생 프로필 응답 스키마."""
    id: int
    preferences: str

    class Config:
        from_attributes = True


class ScoreCreate(BaseModel):
    """학생 점수 생성 요청 스키마."""
    evaluation_item_id: int
    score: float


class ScoreResponse(ScoreCreate):
    """학생 점수 응답 스키마."""
    id: int

    class Config:
        from_attributes = True


class CourseCreate(BaseModel):
    """과목 생성 요청 스키마."""
    name: str
    course_code: str
    total_students: Optional[int] = 99


class CourseResponse(CourseCreate):
    """과목 응답 스키마."""
    id: int

    class Config:
        from_attributes = True


class EvaluationItemCreate(BaseModel):
    """평가 항목 생성 요청 스키마."""
    course_id: int
    name: str
    weight: int
    my_score: Optional[float] = None
    is_submitted: Optional[bool] = False


class EvaluationItemResponse(EvaluationItemCreate):
    """평가 항목 응답 스키마."""
    id: int

    class Config:
        from_attributes = True


class CourseReviewCreate(BaseModel):
    """과목 수강평 생성 요청 스키마."""
    course_id: int
    content: str


class CourseReviewResponse(CourseReviewCreate):
    """과목 수강평 응답 스키마."""
    id: int

    class Config:
        from_attributes = True


class HistogramPredictRequest(BaseModel):
    """히스토그램 예측 요청 스키마."""
    evaluation_item_id: int


class HistogramPredictResponse(BaseModel):
    """히스토그램 예측 응답 스키마."""
    evaluation_item_id: int
    histogram: dict
    num_samples: int
    sample_scores: List[float]
    total_students: Optional[int] = None


class ReviewAnalysisResponse(BaseModel):
    """과목 수강평 분석 및 학습 조언 응답 스키마."""
    assignment_difficulty: int = Field(..., description="과제 난이도 (1~5)")
    exam_difficulty: int = Field(..., description="시험 난이도 (1~5)")
    summary: str = Field(..., description="시험/과제 공통 언급 요약")
    advice: str = Field(..., description="목표 성적 달성 조언")


class SemesterPlanItem(BaseModel):
    """학기 계획 항목 스키마."""
    course_index: int = Field(..., description="입력된 과목 순서 (1부터 시작)")
    effort_percent: int = Field(..., description="투자해야 할 노력 비율 (0~100)")


class SemesterPlanResponse(BaseModel):
    """학기 전체 학습 계획 응답 스키마."""
    courses: List[SemesterPlanItem]
    overall_advice: str = Field(..., description="전체 학기 운영을 위한 1-2문장 조언")


class CumulativeHistogramResponse(BaseModel):
    """과목별 누적 히스토그램 응답 스키마."""
    course_id: int
    cumulative_histogram: dict
    total_weight: int
    evaluation_items: List[dict]


app = FastAPI(
    title="Smart Learning Strategy & Grade Toolkit API",
    version="1.0.0",
    description="SetTransformer 딥러닝 모델 기반 성적 분포 예측 및 OpenAI 기반 학습 전략 추천 시스템"
)

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

ml_predictor = None


@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 ML 모델을 로드합니다."""
    global ml_predictor
    try:
        from ML.model_loader import HistogramPredictor
        model_path = os.path.join(BASE_DIR, "ML", "best_model_nnj359uw.pt")
        ml_predictor = HistogramPredictor(model_path=model_path)
        print("✓ ML model loaded successfully")
    except:
        print("⚠ ML module skipped.")
        ml_predictor = None


def get_db():
    """데이터베이스 세션 의존성 주입 함수."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health", tags=["System"])
async def health_check():
    """서버 상태 확인 엔드포인트."""
    return {"status": "healthy"}


@app.get("/dummy-histo", tags=["Development"])
async def get_dummy_histogram():
    """테스트용 더미 히스토그램 데이터를 반환합니다."""
    return {"0-10" : 5, "10-20": 15, "20-30": 25, "30-40": 10, "40-50": 8, "50-60": 12, "60-70": 7, "70-80": 3,
            "80-90": 1, "90-100": 0}


@app.put("/student-profile", response_model=StudentProfileResponse, tags=["Student Profile"])
async def update_student_profile(profile: StudentProfileCreate, db: Session = Depends(get_db)):
    """
    학생 프로필을 업데이트합니다. ID 1번 학생의 프로필을 항상 업데이트합니다.
    """
    student_id = 1
    existing_profile = db.query(StudentProfileModel).filter(StudentProfileModel.id == student_id).first()

    if existing_profile:
        # 기존 프로필 업데이트
        existing_profile.preferences = profile.preferences
        db.commit()
        db.refresh(existing_profile)
        return existing_profile
    else:
        # 프로필이 없으면 새로 생성
        new_profile = StudentProfileModel(id=student_id, preferences=profile.preferences)
        db.add(new_profile)
        db.commit()
        db.refresh(new_profile)
        return new_profile


@app.get("/student-profile", response_model=StudentProfileResponse, tags=["Student Profile"])
async def get_student_profile(db: Session = Depends(get_db)):
    """
    학생 프로필을 조회합니다. ID 1번 학생의 프로필을 반환합니다.
    """
    student_id = 1
    profile = db.query(StudentProfileModel).filter(StudentProfileModel.id == student_id).first()

    if not profile:
        raise HTTPException(status_code=404, detail="학생 프로필이 없습니다.")

    return profile


@app.post("/courses", response_model=CourseResponse, tags=["Courses"])
async def create_course(course: CourseCreate, db: Session = Depends(get_db)):
    """새로운 과목을 생성합니다."""
    new_course = CourseModel(**course.dict())
    db.add(new_course)
    db.commit()
    db.refresh(new_course)
    return new_course


@app.get("/courses", response_model=List[CourseResponse], tags=["Courses"])
async def get_all_courses(db: Session = Depends(get_db)):
    """모든 과목 목록을 조회합니다."""
    return db.query(CourseModel).all()


@app.post("/evaluation-items", response_model=EvaluationItemResponse, tags=["Evaluation Items"])
async def create_evaluation_item(item: EvaluationItemCreate, db: Session = Depends(get_db)):
    """새로운 평가 항목(과제, 시험 등)을 생성합니다."""
    new_item = EvaluationItemModel(**item.dict())
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return new_item


@app.get("/evaluation-items", response_model=List[EvaluationItemResponse], tags=["Evaluation Items"])
async def get_all_evaluation_items(db: Session = Depends(get_db)):
    """모든 평가 항목 목록을 조회합니다."""
    return db.query(EvaluationItemModel).all()


@app.post("/course-reviews", response_model=CourseReviewResponse, tags=["Course Reviews"])
async def create_course_review(review: CourseReviewCreate, db: Session = Depends(get_db)):
    """새로운 과목 수강평을 생성합니다."""
    new_review = CourseReviewModel(**review.dict())
    db.add(new_review)
    db.commit()
    db.refresh(new_review)
    return new_review


@app.get("/course-reviews", response_model=List[CourseReviewResponse], tags=["Course Reviews"])
async def get_all_course_reviews(db: Session = Depends(get_db)):
    """모든 과목 수강평 목록을 조회합니다."""
    return db.query(CourseReviewModel).all()


@app.post("/other-student-scores", response_model=ScoreResponse, tags=["Other Student Scores"])
async def create_other_score(score_data: ScoreCreate, db: Session = Depends(get_db)):
    """새로운 학생 점수 데이터를 생성합니다."""
    new_score = OtherStudentScoreModel(**score_data.dict())
    db.add(new_score)
    db.commit()
    db.refresh(new_score)
    return new_score


@app.get("/other-student-scores", response_model=List[ScoreResponse], tags=["Other Student Scores"])
async def get_other_scores(item_id: Optional[int] = None, db: Session = Depends(get_db)):
    """학생 점수 목록을 조회합니다. 평가 항목 ID로 필터링할 수 있습니다."""
    query = db.query(OtherStudentScoreModel)
    if item_id:
        query = query.filter(OtherStudentScoreModel.evaluation_item_id == item_id)
    return query.all()


@app.get("/predict-histogram", response_model=HistogramPredictResponse, tags=["ML Prediction"])
def predict_histogram(evaluation_item_id: int, db: Session = Depends(get_db)):
    """
    평가 항목의 샘플 점수로부터 전체 학급의 성적 분포 히스토그램을 예측합니다.
    SetTransformer 모델을 사용하여 10개 구간(0-10, 10-20, ..., 90-100)의 분포를 예측합니다.
    """
    if ml_predictor is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    scores = db.query(OtherStudentScoreModel).filter(
            OtherStudentScoreModel.evaluation_item_id == evaluation_item_id).all()
    if not scores:
        raise HTTPException(status_code=404, detail="No scores found")
    score_values = [s.score for s in scores]

    total = None
    item = db.query(EvaluationItemModel).filter(EvaluationItemModel.id == evaluation_item_id).first()
    if item:
        course = db.query(CourseModel).filter(CourseModel.id == item.course_id).first()
        if course and course.total_students:
            total = course.total_students
    if total is None:
        total = 99

    try:
        histogram = ml_predictor.predict(score_values, total_students=total)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return HistogramPredictResponse(
            evaluation_item_id=evaluation_item_id,
            histogram=histogram,
            num_samples=len(score_values),
            sample_scores=score_values,
            total_students=total
    )


@app.get("/courses/{course_id}/advice", response_model=ReviewAnalysisResponse, tags=["AI Advice"])
def get_course_advice(course_id: int, objective_grade: str, db: Session = Depends(get_db)):
    """
    과목의 수강평을 분석하여 목표 성적 달성을 위한 학습 조언을 제공합니다.
    OpenAI API를 사용하여 과제 및 시험 난이도 분석과 학습 전략을 생성합니다.
    """
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI API Key missing")
    reviews = db.query(CourseReviewModel).filter(CourseReviewModel.course_id == course_id).all()
    if not reviews:
        raise HTTPException(status_code=404, detail="리뷰 데이터가 없습니다.")
    course_reviews_str = "\n".join([f"- {r.content}" for r in reviews])

    import json
    import re

    try:
        response = openai_client.responses.create(
                model="gpt-5-mini",
                input=f"""
            목표성적: {objective_grade}
            이건 이전 수강자들의 강의평 입니다. 각 강의평은 과목 ID와 함께 주어집니다.
            강의평에서 과제의 난이도와 관련한 정보를 추출해서 난이도를 정수형으로 추출해서 반환해주세요.
            강의평에서 시험의 난이도와 관련한 정보를 추출해서 난이도를 정수형으로 추출해서 반환해주세요.
            난이도는 1부터 5까지의 정수로 추출해주세요.
            1: 매우 쉬움
            2: 쉬움
            3: 보통
            4: 어려움
            5: 매우 어려움
            시험과 과제와 관련해 공통된 언급들은 정리해서 1-2문장으로 정리해주세요.
            정리한 내용을 바탕으로 각 과제와 시험 중 어느 곳에 집중해야 할지 1-2문장으로 정리해서 조언형식으로 반환해주세요.
            이 때 조언에는 과제와 시험 난이도 특성을 근거로 각 과제와 시험 중 어느 곳에 집중해야 할지 정리해주세요.
            조언에는 목표성적에 따라 각 과제와 시험 중 어느 곳에 집중해야 할지 정리해주세요.
            만약 하지 않아도 되는 과제가 있다면 그 과제는 하지 않아도 된다고 정리해주세요.
            만약 그런 과제가 없다면 이 부분은 언급하지 말아주세요.
            각 과제의 성적 비중이 어느 정도인지에 따라 각 과제와 시험 중 어느 곳에 집중해야 할지 정리해주세요.
            비중이 크고 작은 것을 판단할 때에는 강의평에 언급된 경우에 대한 내용을 참고해주세요.
            
            강의평:
            {course_reviews_str}
            """,
                text={
                        "verbosity": "low",
                        "format"   : {
                                "type"  : "json_schema",
                                "name"  : "course_advice",
                                "schema": {
                                        "type"                : "object",
                                        "properties"          : {
                                                "assignment_difficulty": {"type": "integer"},
                                                "exam_difficulty"      : {"type": "integer"},
                                                "summary"              : {"type": "string"},
                                                "advice"               : {"type": "string"}
                                        },
                                        "required"            : ["assignment_difficulty", "exam_difficulty", "summary",
                                                                 "advice"],
                                        "additionalProperties": False
                                }
                        }
                },
                reasoning={"effort": "minimal"},
        )

        text = response.output_text.strip()

        # Try to extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Try to find JSON object in the text
        if not text.startswith('{'):
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)

        result = json.loads(text)
        return ReviewAnalysisResponse(**result)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500,
                            detail=f"JSON Parse Error: {str(e)} - Response: {response.output_text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")


@app.get("/semester-advice", response_model=SemesterPlanResponse, tags=["AI Advice"])
def get_semester_advice(
        course_ids: List[int] = Query(..., description="수강할 과목 ID 리스트 (예: 1, 2, 3)"),
        target_grades: List[str] = Query(..., description="각 과목의 목표 성적 (예: A+, A, B+)"),
        db: Session = Depends(get_db)
):
    """
    여러 과목의 리뷰를 종합하여 전체 학기 공부 비중(%)과 전략을 짜줍니다.
    OpenAI API를 사용하여 각 과목별 노력 배분 비율과 전체 학기 조언을 생성합니다.

    - course_ids의 순서와 target_grades의 순서는 일치해야 합니다.
    - 예: /semester-advice?course_ids=1&course_ids=2&target_grades=A+&target_grades=B0
    """
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI API Key missing")

    if len(course_ids) != len(target_grades):
        raise HTTPException(status_code=400, detail="과목 수와 목표 성적 수가 일치해야 합니다.")

    combined_reviews_text = ""

    for idx, (cid, grade) in enumerate(zip(course_ids, target_grades)):
        course = db.query(CourseModel).filter(CourseModel.id == cid).first()
        if not course:
            continue

        reviews = db.query(CourseReviewModel).filter(CourseReviewModel.course_id == cid).all()
        review_texts = "\n".join([f"- {r.content}" for r in reviews]) if reviews else "리뷰 없음"

        combined_reviews_text += f"\n[과목 {idx + 1}: {course.name} (목표: {grade})]\n{review_texts}\n"

    if not combined_reviews_text:
        raise HTTPException(status_code=404, detail="선택한 과목들에 대한 리뷰 데이터가 없습니다.")

    import json
    import re

    try:
        response = openai_client.responses.create(
                model="gpt-5-mini",
                input=f"""
            너는 학습 계획을 설계하는 조교이다.

            반드시 아래 JSON Schema를 만족하는 JSON 한 개만 출력해야 한다.
            - JSON 이외의 텍스트(설명, 마크다운 등)는 절대 출력하지 마라.

            과목 3개에 대한 리뷰가 주어지며,
            선택한 과목들의 목표 성적은 순서대로: {", ".join(target_grades)}
            (하지만 목표 성적은 JSON 출력 형식에는 포함하지 않는다.)

            아래 규칙에 따라 JSON을 생성하라.

            [규칙]
            1) 각 과목에 대해 effort_percent(0~100 정수)를 정하라.
            2) effort_percent들의 합은 반드시 100이 되어야 한다.
            3) courses 배열의 course_index는 1, 2, 3 중 하나로 고정한다.
            4) 전체 학기에 대한 조언(overall_advice)은 1~2문장으로만 작성한다.

            --------- 수강평 시작 ---------
            {combined_reviews_text}
            --------- 수강평 끝 ---------
            """,
                text={
                        "verbosity": "low",
                        "format"   : {
                                "type"  : "json_schema",
                                "name"  : "semester_plan",
                                "schema": {
                                        "type"                : "object",
                                        "properties"          : {
                                                "courses"       : {
                                                        "type" : "array",
                                                        "items": {
                                                                "type"                : "object",
                                                                "properties"          : {
                                                                        "course_index"  : {"type": "integer"},
                                                                        "effort_percent": {"type": "integer"}
                                                                },
                                                                "required"            : ["course_index",
                                                                                         "effort_percent"],
                                                                "additionalProperties": False
                                                        }
                                                },
                                                "overall_advice": {"type": "string"}
                                        },
                                        "required"            : ["courses", "overall_advice"],
                                        "additionalProperties": False
                                }
                        }
                },
                reasoning={"effort": "minimal"},
        )

        text = response.output_text.strip()

        # Try to extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Try to find JSON object in the text
        if not text.startswith('{'):
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)

        result = json.loads(text)
        return SemesterPlanResponse(**result)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500,
                            detail=f"JSON Parse Error: {str(e)} - Response: {response.output_text[:200]}")
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI 분석 중 오류가 발생했습니다.")


@app.get("/courses/{course_id}/cumulative-histogram", response_model=CumulativeHistogramResponse, tags=["ML Prediction"])
def get_cumulative_histogram(course_id: int, db: Session = Depends(get_db)):
    """
    과목의 모든 평가 항목들의 히스토그램을 가중치에 따라 누적합니다.
    각 evaluation_item의 히스토그램에 weight를 곱한 뒤 모두 합산하여 최종 성적 분포를 예측합니다.

    예: 과제1(20%) + 과제2(20%) + 중간고사(30%) + 기말고사(30%)
    """
    if ml_predictor is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")

    items = db.query(EvaluationItemModel).filter(EvaluationItemModel.course_id == course_id).all()
    if not items:
        raise HTTPException(status_code=404, detail="해당 과목의 평가 항목이 없습니다.")

    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    total_students = course.total_students if course and course.total_students else 99

    cumulative_histogram = {
        "0-10": 0.0, "10-20": 0.0, "20-30": 0.0, "30-40": 0.0, "40-50": 0.0,
        "50-60": 0.0, "60-70": 0.0, "70-80": 0.0, "80-90": 0.0, "90-100": 0.0
    }

    total_weight = sum(item.weight for item in items)
    evaluation_items_info = []

    for item in items:
        scores = db.query(OtherStudentScoreModel).filter(
            OtherStudentScoreModel.evaluation_item_id == item.id
        ).all()

        if not scores:
            evaluation_items_info.append({
                "id": item.id,
                "name": item.name,
                "weight": item.weight,
                "histogram": None,
                "note": "점수 데이터 없음"
            })
            continue

        score_values = [s.score for s in scores]

        try:
            histogram = ml_predictor.predict(score_values, total_students=total_students)

            weight_ratio = item.weight / 100.0
            for bin_range, count in histogram.items():
                if bin_range in cumulative_histogram:
                    cumulative_histogram[bin_range] += count * weight_ratio

            evaluation_items_info.append({
                "id": item.id,
                "name": item.name,
                "weight": item.weight,
                "histogram": histogram,
                "num_samples": len(score_values)
            })
        except Exception as e:
            evaluation_items_info.append({
                "id": item.id,
                "name": item.name,
                "weight": item.weight,
                "histogram": None,
                "error": str(e)
            })

    return CumulativeHistogramResponse(
        course_id=course_id,
        cumulative_histogram=cumulative_histogram,
        total_weight=total_weight,
        evaluation_items=evaluation_items_info
    )

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional

# ---------------------------------------------------------
# 1. Database Setup
# ---------------------------------------------------------
SQLALCHEMY_DATABASE_URL = "sqlite:///./hackathon.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------------------------------------------------
# 2. Database Models (SQLAlchemy)
# ---------------------------------------------------------

# 1. 학생 성향
class StudentProfileModel(Base):
    __tablename__ = "student_profile"
    id = Column(Integer, primary_key=True, index=True)
    preferences = Column(String)

# 2. 타 학생 점수
class OtherStudentScoreModel(Base):
    __tablename__ = "other_student_scores"
    id = Column(Integer, primary_key=True, index=True)
    evaluation_item_id = Column(Integer, index=True)
    score = Column(Float)

# 3. 강의 정보 (New)
class CourseModel(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    division = Column(String, nullable=True)
    grading_type = Column(String, default="RELATIVE")

# 4. 평가 항목 (New)
class EvaluationItemModel(Base):
    __tablename__ = "evaluation_items"
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, index=True)
    name = Column(String)
    weight = Column(Integer)
    my_score = Column(Float, nullable=True)
    is_submitted = Column(Boolean, default=False)

# 5. 강의평 (New)
class CourseReviewModel(Base):
    __tablename__ = "course_reviews"
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, index=True)
    content = Column(String)
    generosity = Column(Integer)

# 테이블 생성 (이미 있으면 건너뜀)
Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------
# 3. Pydantic Schemas (API 입출력 검증)
# ---------------------------------------------------------

# [Student Profile]
class StudentProfileCreate(BaseModel):
    preferences: str

class StudentProfileResponse(BaseModel):
    id: int
    preferences: str
    class Config:
        from_attributes = True

# [Other Student Scores]
class ScoreCreate(BaseModel):
    evaluation_item_id: int
    score: float

class ScoreResponse(ScoreCreate):
    id: int
    class Config:
        from_attributes = True

# [Courses] (New)
class CourseCreate(BaseModel):
    name: str
    division: Optional[str] = None
    grading_type: Optional[str] = "RELATIVE"

class CourseResponse(CourseCreate):
    id: int
    class Config:
        from_attributes = True

# [Evaluation Items] (New)
class EvaluationItemCreate(BaseModel):
    course_id: int
    name: str
    weight: int
    my_score: Optional[float] = None
    is_submitted: Optional[bool] = False

class EvaluationItemResponse(EvaluationItemCreate):
    id: int
    class Config:
        from_attributes = True

# [Course Reviews] (New)
class CourseReviewCreate(BaseModel):
    course_id: int
    content: str
    generosity: int

class CourseReviewResponse(CourseReviewCreate):
    id: int
    class Config:
        from_attributes = True


# ---------------------------------------------------------
# 4. FastAPI App
# ---------------------------------------------------------
app = FastAPI(title="Hackathon API", version="1.0.0")

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/dumy-histo")
async def get_dummy_histogram():
    dummy_histogram = {
            "0-10"  : 5,
            "10-20" : 15,
            "20-30" : 25,
            "30-40" : 10,
            "40-50" : 8,
            "50-60" : 12,
            "60-70" : 7,
            "70-80" : 3,
            "80-90" : 1,
            "90-100": 0,
    }
    return dummy_histogram
# ---------------------------------------------------------
# 5. API Endpoints
# ---------------------------------------------------------

# --- 1. Student Profile ---
@app.post("/student-profile", response_model=StudentProfileResponse, tags=["Student Profile"])
async def create_student_profile(profile: StudentProfileCreate, db: Session = Depends(get_db)):
    new_profile = StudentProfileModel(preferences=profile.preferences)
    db.add(new_profile)
    db.commit()
    db.refresh(new_profile)
    return new_profile

@app.get("/student-profile", response_model=List[StudentProfileResponse], tags=["Student Profile"])
async def get_all_student_profiles(db: Session = Depends(get_db)):
    return db.query(StudentProfileModel).all()


# --- 2. Courses (New) ---
@app.post("/courses", response_model=CourseResponse, tags=["Courses"])
async def create_course(course: CourseCreate, db: Session = Depends(get_db)):
    new_course = CourseModel(
        name=course.name,
        division=course.division,
        grading_type=course.grading_type
    )
    db.add(new_course)
    db.commit()
    db.refresh(new_course)
    return new_course

@app.get("/courses", response_model=List[CourseResponse], tags=["Courses"])
async def get_all_courses(db: Session = Depends(get_db)):
    return db.query(CourseModel).all()


# --- 3. Evaluation Items (New) ---
@app.post("/evaluation-items", response_model=EvaluationItemResponse, tags=["Evaluation Items"])
async def create_evaluation_item(item: EvaluationItemCreate, db: Session = Depends(get_db)):
    new_item = EvaluationItemModel(
        course_id=item.course_id,
        name=item.name,
        weight=item.weight,
        my_score=item.my_score,
        is_submitted=item.is_submitted
    )
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return new_item

@app.get("/evaluation-items", response_model=List[EvaluationItemResponse], tags=["Evaluation Items"])
async def get_all_evaluation_items(db: Session = Depends(get_db)):
    return db.query(EvaluationItemModel).all()


# --- 4. Course Reviews (New) ---
@app.post("/course-reviews", response_model=CourseReviewResponse, tags=["Course Reviews"])
async def create_course_review(review: CourseReviewCreate, db: Session = Depends(get_db)):
    new_review = CourseReviewModel(
        course_id=review.course_id,
        content=review.content,
        generosity=review.generosity
    )
    db.add(new_review)
    db.commit()
    db.refresh(new_review)
    return new_review

@app.get("/course-reviews", response_model=List[CourseReviewResponse], tags=["Course Reviews"])
async def get_all_course_reviews(db: Session = Depends(get_db)):
    return db.query(CourseReviewModel).all()


# --- 5. Other Student Scores ---
@app.post("/other-student-scores", response_model=ScoreResponse, tags=["Other Student Scores"])
async def create_other_score(score_data: ScoreCreate, db: Session = Depends(get_db)):
    new_score = OtherStudentScoreModel(
        evaluation_item_id=score_data.evaluation_item_id,
        score=score_data.score
    )
    db.add(new_score)
    db.commit()
    db.refresh(new_score)
    return new_score

@app.get("/other-student-scores", response_model=List[ScoreResponse], tags=["Other Student Scores"])
async def get_other_scores(item_id: Optional[int] = None, db: Session = Depends(get_db)):
    query = db.query(OtherStudentScoreModel)
    if item_id:
        query = query.filter(OtherStudentScoreModel.evaluation_item_id == item_id)
    return query.all()

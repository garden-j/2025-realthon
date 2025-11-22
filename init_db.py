import sqlite3
import os

# DB 파일 이름 설정
DB_NAME = "hackathon.db"

def init_db():
    # 1. 기존 DB 파일 삭제 (스키마 변경 적용을 위해 필수!)
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        print(f"🗑️ 기존 {DB_NAME} 파일을 삭제했습니다. (스키마 업데이트를 위해)")

    # DB 연결
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # FK 활성화
    cursor.execute("PRAGMA foreign_keys = ON;")

    print("🛠️ 테이블 생성을 시작합니다...")

    # 1. Student Profile
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS student_profile (
        id INTEGER PRIMARY KEY,
        preferences TEXT
    )
    ''')

    # 2. Courses
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS courses (
        id INTEGER PRIMARY KEY, 
        name TEXT NOT NULL,
        division TEXT,
        grading_type TEXT DEFAULT 'RELATIVE'
    )
    ''')

    # 3. Evaluation Items
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS evaluation_items (
        id INTEGER PRIMARY KEY,
        course_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        weight INTEGER NOT NULL,
        my_score REAL DEFAULT NULL,
        is_submitted BOOLEAN DEFAULT 0,
        FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE
    )
    ''')

    # 4. Other Student Scores
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS other_student_scores (
        id INTEGER PRIMARY KEY,
        evaluation_item_id INTEGER NOT NULL,
        score REAL NOT NULL,
        FOREIGN KEY (evaluation_item_id) REFERENCES evaluation_items(id) ON DELETE CASCADE
    )
    ''')

    # 5. Course Reviews (★ 수정됨: generosity 컬럼)
    # 0: 짜게 줌, 1: 보통, 2: 후하게 줌
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS course_reviews (
        id INTEGER PRIMARY KEY,
        course_id INTEGER NOT NULL,
        content TEXT,
        generosity INTEGER CHECK(generosity IN (0, 1, 2)), 
        FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE
    )
    ''')

    print("✅ 테이블 생성 완료!")
    
    # ---------------------------------------------------------
    # 🧪 더미 데이터 삽입
    # ---------------------------------------------------------
    print("📥 더미 데이터 입력 중...")

    # 1. 내 정보
    cursor.execute("INSERT INTO student_profile (id, preferences) VALUES (?, ?)", 
                   (1, "암기형, 객관식 선호, 과제보다 시험 선호, 목표: B0"))

    # 2. 강의
    cursor.execute("INSERT INTO courses (name, division, grading_type) VALUES (?, ?, ?)", 
                   ("운영체제", "A반", "RELATIVE"))
    course_id = cursor.lastrowid 

    # 3. 평가 항목
    items = [
        (course_id, "중간고사", 30, 90.0, 1), 
        (course_id, "기말고사", 30, None, 0), 
        (course_id, "과제 1", 20, 100.0, 1), 
        (course_id, "과제 2", 20, None, 0)
    ]
    cursor.executemany("INSERT INTO evaluation_items (course_id, name, weight, my_score, is_submitted) VALUES (?, ?, ?, ?, ?)", items)

    # 4. 타 학생 점수
    other_scores = [
        (1, 85.5), (1, 92.0), (1, 40.0), (1, 78.0), (1, 60.0), # 중간고사
        (3, 100.0), (3, 95.0), (3, 88.0), (3, 100.0), (3, 70.0) # 과제 1
    ]
    cursor.executemany("INSERT INTO other_student_scores (evaluation_item_id, score) VALUES (?, ?)", other_scores)

    # 5. 강의평 (★ 데이터도 변경됨)
    reviews = [
        # (course_id, content, generosity)
        # 0: 점수 안 줌, 2: 점수 잘 줌
        (course_id, "과제 2번은 교수님이 코드를 꼼꼼하게 봐서 감점이 많아요.", 0), 
        (course_id, "중간고사는 부분 점수를 엄청 후하게 주십니다. 백지 아니면 됨.", 2) 
    ]
    cursor.executemany("INSERT INTO course_reviews (course_id, content, generosity) VALUES (?, ?, ?)", reviews)

    conn.commit()
    conn.close()
    print(f"🎉 '{DB_NAME}' 파일 재생성 완료! (generosity 컬럼 적용됨)")

if __name__ == "__main__":
    init_db()
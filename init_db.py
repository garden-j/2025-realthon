import sqlite3
import os

DB_NAME = "hackathon.db"


def init_db():
    # 기존 파일 삭제 (스키마 변경 적용)
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        print(f"🗑️ 기존 {DB_NAME} 파일을 삭제했습니다.")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    print("🛠️ 테이블 생성을 시작합니다...")

    # 1. Student Profile
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS student_profile
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY,
                       preferences
                       TEXT
                   )
                   ''')

    # 2. Courses (★ 수정됨: 학수번호만 남김)
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS courses
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY,
                       name
                       TEXT
                       NOT
                       NULL,   -- 강의명 (예: 운영체제)
                       course_code
                       TEXT,   -- 학수번호 (예: COSE341) - 식별자
                       total_students
                       INTEGER -- 수강생 수
                   )
                   ''')

    # 3. Evaluation Items
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS evaluation_items
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY,
                       course_id
                       INTEGER
                       NOT
                       NULL,
                       name
                       TEXT
                       NOT
                       NULL,
                       weight
                       INTEGER
                       NOT
                       NULL,
                       my_score
                       REAL
                       DEFAULT
                       NULL,
                       is_submitted
                       BOOLEAN
                       DEFAULT
                       0,
                       FOREIGN
                       KEY
                   (
                       course_id
                   ) REFERENCES courses
                   (
                       id
                   ) ON DELETE CASCADE
                       )
                   ''')

    # 4. Other Student Scores
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS other_student_scores
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY,
                       evaluation_item_id
                       INTEGER
                       NOT
                       NULL,
                       score
                       REAL
                       NOT
                       NULL,
                       FOREIGN
                       KEY
                   (
                       evaluation_item_id
                   ) REFERENCES evaluation_items
                   (
                       id
                   ) ON DELETE CASCADE
                       )
                   ''')

    # 5. Course Reviews
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS course_reviews
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY,
                       course_id
                       INTEGER
                       NOT
                       NULL,
                       content
                       TEXT,
                       FOREIGN
                       KEY
                   (
                       course_id
                   ) REFERENCES courses
                   (
                       id
                   ) ON DELETE CASCADE
                       )
                   ''')

    conn.commit()
    conn.close()
    print(f"🎉 '{DB_NAME}' 파일 생성 및 스키마 업데이트 완료! (심플 버전)")


if __name__ == "__main__":
    init_db()

import sqlite3
import random
import os

# DB 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hackathon.db")


def fill_data():
    if not os.path.exists(DB_PATH):
        print(f"❌ 오류: '{DB_PATH}' 파일을 찾을 수 없습니다. init_db.py를 먼저 실행해주세요.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("🔄 데이터 채우기 시작...")

    # ---------------------------------------------------------
    # 1. 기존 데이터 초기화 (중복 방지)
    # ---------------------------------------------------------
    # courses, course_reviews는 건드리지 않고 나머지 3개만 비웁니다.
    cursor.execute("DELETE FROM other_student_scores")
    cursor.execute("DELETE FROM evaluation_items")
    cursor.execute("DELETE FROM student_profile")
    print("🧹 기존 평가/성적/프로필 데이터를 초기화했습니다.")

    # ---------------------------------------------------------
    # 2. Student Profile 생성
    # ---------------------------------------------------------
    profile_text = "암기형, 객관식 선호, 과제보다 시험 선호"
    cursor.execute("INSERT INTO student_profile (preferences) VALUES (?)", (profile_text,))
    print(f"👤 학생 프로필 생성 완료: {profile_text}")

    # ---------------------------------------------------------
    # 3. Evaluation Items & Scores 생성
    # ---------------------------------------------------------
    # 현재 DB에 있는 모든 강의 ID 조회
    cursor.execute("SELECT id, name FROM courses")
    courses = cursor.fetchall()

    if not courses:
        print("⚠ 경고: courses 테이블이 비어있습니다. import_csv.py를 먼저 실행하세요.")
        return

    total_items = 0
    total_scores = 0

    for course_id, course_name in courses:
        # 각 강의마다 만들 평가 항목 리스트 (이름, 배점)
        # 예: 과제1(20%), 과제2(20%), 중간고사(30%) -> 총 70% (기말은 나중에 본다고 가정)
        items_to_create = [
                ("과제 1", 20),
                ("과제 2", 20),
                ("중간고사", 30)
        ]

        for item_name, weight in items_to_create:
            # 3-1. 평가 항목(Evaluation Item) Insert
            # my_score는 80로 설정, 필요하면 값 넣어도 됨
            cursor.execute('''
                           INSERT INTO evaluation_items (course_id, name, weight, my_score, is_submitted)
                           VALUES (?, ?, ?, 80, 1)
                           ''', (course_id, item_name, weight))

            # 방금 만든 항목의 ID 가져오기
            item_id = cursor.lastrowid
            total_items += 1

            # 3-2. 해당 항목에 대한 타 학생 점수(Other Student Scores) 10개 생성
            # 점수는 60점 ~ 100점 사이 랜덤 (중간고사는 좀 더 분포가 넓게)
            for _ in range(10):
                if "중간고사" in item_name:
                    score = round(random.uniform(40.0, 100.0), 1)  # 시험은 점수 편차가 큼
                else:
                    score = round(random.uniform(70.0, 100.0), 1)  # 과제는 보통 점수가 높음

                cursor.execute('''
                               INSERT INTO other_student_scores (evaluation_item_id, score)
                               VALUES (?, ?)
                               ''', (item_id, score))
                total_scores += 1

    conn.commit()
    conn.close()

    print("\n✅ 데이터 주입 완료!")
    print(f"   - 대상 강의 수: {len(courses)}개")
    print(f"   - 생성된 평가 항목: {total_items}개")
    print(f"   - 생성된 학생 점수: {total_scores}개")


if __name__ == "__main__":
    fill_data()

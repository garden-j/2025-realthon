import sqlite3

# DB 파일 연결
DB_NAME = "hackathon.db"
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

print("과제 1(ID: 3) 점수 데이터를 추가합니다...")

# 추가할 점수 데이터 (evaluation_item_id=3, score)
new_scores = [
    (3, 100.0), 
    (3, 95.0), 
    (3, 88.0), 
    (3, 100.0), 
    (3, 70.0)
]

# 데이터 삽입
cursor.executemany("INSERT INTO other_student_scores (evaluation_item_id, score) VALUES (?, ?)", new_scores)

conn.commit()
conn.close()

print(f"✅ {len(new_scores)}명의 점수가 성공적으로 추가되었습니다!")
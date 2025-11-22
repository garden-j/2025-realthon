from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, Field  # ← 새로 추가


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

df = pd.read_csv("../crawling/cose341_klue_reviews.csv")
reviews = df["review"].tolist()

course_reviews = []

for review in reviews:
    course_reviews.append(
        {
            "course_id": 1,
            "content": review,
            "generosity": 0,
            "id": len(course_reviews) + 1
        }
    )

course_reviews_str = "\n".join([f"{review['content']}" for review in course_reviews])

objective_grade = "B+"


response = client.responses.create(
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
    {course_reviews_str}
    """,
    text={
        "verbosity": "low",
    },
    reasoning={"effort": "low"},
)

print(response.output_text)

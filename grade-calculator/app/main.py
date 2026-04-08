from fastapi import FastAPI, HTTPException
from app.schema import GradeInput, GradeOutput

app = FastAPI(
    title="Student Grade Calculator",
    description="via fastapi",
    version = "1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok"}



@app.post("/calculate_grade", response_model=GradeOutput)
def calculate_grade(grade_input: GradeInput):
    
    if len(grade_input.scores) == 0:
        raise HTTPException(status_code=400, detail="Scores list cannot be empty.")
    
    average_score = sum(grade_input.scores) / len(grade_input.scores)
    
    if average_score >= 90:
        letter_grade = 'A'
    elif average_score >= 80:
        letter_grade = 'B'
    elif average_score >= 70:
        letter_grade = 'C'
    elif average_score >= 60:
        letter_grade = 'D'
    else:
        letter_grade = 'F'
    
    passed = average_score >= 60
    
    return GradeOutput(
        student_name=grade_input.student_name,
        average=average_score,
        letter_grade=letter_grade,
        passed=passed
    )
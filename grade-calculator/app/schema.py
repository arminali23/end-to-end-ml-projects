from pydantic import BaseModel

class GradeInput(BaseModel):
    student_name: str
    scores : list[float]
    
class GradeOutput(BaseModel):
    student_name: str
    average: float
    letter_grade: str
    passed: bool
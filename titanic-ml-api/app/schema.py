from pydantic import BaseModel
from typing import Literal

class PassengerInput(BaseModel):
    Pclass: int          # 1, 2, or 3
    Sex: Literal["male", "female"]
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: Literal["S", "C", "Q"]

class PredictionOutput(BaseModel):
    survived: bool
    survival_probability: float
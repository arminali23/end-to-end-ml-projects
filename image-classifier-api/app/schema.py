from pydantic import BaseModel

class Prediction(BaseModel):
    label: str
    confidence: float

class ClassificationOutput(BaseModel):
    top_predictions: list[Prediction]
    model: str = "ResNet50-ImageNet"
from pydantic import BaseModel
from typing import List


class Top5Entry(BaseModel):
    cls: str = None
    confidence: float

    class Config:
        # allow "class" key from the dict even though it's a Python keyword
        populate_by_name = True

    @classmethod
    def from_dict(cls, d: dict):
        return cls(cls=d["class"], confidence=d["confidence"])


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    plant: str
    disease: str
    is_healthy: bool
    plant_filter_used: bool
    top5: List[dict]


class HealthResponse(BaseModel):
    status: str
    model: str
    num_classes: int
    input_size: str
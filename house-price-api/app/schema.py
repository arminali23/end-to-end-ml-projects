from pydantic import BaseModel, Field

class HouseInput(BaseModel):
    crim:    float = Field(..., description="Per capita crime rate")
    zn:      float = Field(..., description="Proportion of residential land zoned")
    indus:   float = Field(..., description="Proportion of non-retail business acres")
    chas:    int   = Field(..., description="Charles River dummy variable (0 or 1)")
    nox:     float = Field(..., description="Nitric oxide concentration")
    rm:      float = Field(..., description="Average number of rooms per dwelling")
    age:     float = Field(..., description="Proportion of owner-occupied units built prior to 1940")
    dis:     float = Field(..., description="Weighted distances to employment centres")
    rad:     int   = Field(..., description="Index of accessibility to radial highways")
    tax:     float = Field(..., description="Full-value property-tax rate per $10,000")
    ptratio: float = Field(..., description="Pupil-teacher ratio by town")
    b:       float = Field(..., description="1000(Bk - 0.63)^2 where Bk is proportion of Black residents")
    lstat:   float = Field(..., description="Percentage of lower-status population")

class PredictionOutput(BaseModel):
    predicted_price_usd: float
    predicted_price_k:   float
    model_used:          str
    r2_score:            float
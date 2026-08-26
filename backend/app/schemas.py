from pydantic import BaseModel, Field


class HealthData(BaseModel):

    age: int = Field(
        ...,
        ge=29,
        le=77,
        description="Patient age in years"
    )

    sex: int = Field(
        ...,
        ge=0,
        le=1
    )

    cp: int = Field(
        ...,
        ge=0,
        le=3
    )

    trestbps: float = Field(
        ...,
        ge=94,
        le=200
    )

    chol: float = Field(
        ...,
        ge=126,
        le=564
    )

    fbs: int = Field(
        ...,
        ge=0,
        le=1
    )

    restecg: int = Field(
        ...,
        ge=0,
        le=2
    )

    thalach: float = Field(
        ...,
        ge=71,
        le=202
    )

    exang: int = Field(
        ...,
        ge=0,
        le=1
    )

    oldpeak: float = Field(
        ...,
        ge=0,
        le=6.2
    )

    slope: int = Field(
        ...,
        ge=0,
        le=2
    )

    ca: int = Field(
        ...,
        ge=0,
        le=4
    )

    thal: int = Field(
        ...,
        ge=0,
        le=3
    )
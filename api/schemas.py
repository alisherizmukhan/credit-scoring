"""Request/response schemas. Pydantic rejects malformed input with HTTP 422
before it ever reaches the model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GMSCApplication(BaseModel):
    """Applicant features of the Give Me Some Credit dataset."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "revolving_utilization": 0.35,
                    "age": 42,
                    "past_due_30_59": 0,
                    "debt_ratio": 0.28,
                    "monthly_income": 6500,
                    "open_credit_lines": 7,
                    "times_90_days_late": 0,
                    "real_estate_loans": 1,
                    "past_due_60_89": 0,
                    "dependents": 2,
                }
            ]
        },
    )

    revolving_utilization: float = Field(
        ge=0, description="Balance on credit cards / credit limits (0..1 typically)"
    )
    age: int = Field(ge=18, le=120, description="Age in years")
    past_due_30_59: int = Field(ge=0, le=98, description="Times 30-59 days past due (2y)")
    debt_ratio: float = Field(ge=0, description="Monthly debt payments / monthly income")
    monthly_income: float | None = Field(default=None, ge=0, description="Monthly income")
    open_credit_lines: int = Field(ge=0, description="Open loans and credit lines")
    times_90_days_late: int = Field(ge=0, le=98, description="Times 90+ days late")
    real_estate_loans: int = Field(ge=0, description="Mortgage and real estate loans")
    past_due_60_89: int = Field(ge=0, le=98, description="Times 60-89 days past due (2y)")
    dependents: int | None = Field(default=None, ge=0, le=20, description="Number of dependents")


class HomeCreditApplication(BaseModel):
    """Key Home Credit fields. Any other raw column of application_train / bureau
    aggregates may be sent as well (extra="allow"); missing ones are imputed."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "AMT_INCOME_TOTAL": 202500.0,
                    "AMT_CREDIT": 406597.5,
                    "AMT_ANNUITY": 24700.5,
                    "AMT_GOODS_PRICE": 351000.0,
                    "DAYS_BIRTH": -9461,
                    "DAYS_EMPLOYED": -637,
                    "EXT_SOURCE_2": 0.26,
                    "EXT_SOURCE_3": 0.14,
                    "CODE_GENDER": "M",
                    "NAME_CONTRACT_TYPE": "Cash loans",
                }
            ]
        },
    )

    AMT_INCOME_TOTAL: float = Field(gt=0)
    AMT_CREDIT: float = Field(gt=0)
    AMT_ANNUITY: float | None = Field(default=None, gt=0)
    AMT_GOODS_PRICE: float | None = Field(default=None, gt=0)
    DAYS_BIRTH: int = Field(lt=0, description="Age in days, negative (relative to application)")
    DAYS_EMPLOYED: int | None = None
    EXT_SOURCE_1: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_2: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(default=None, ge=0, le=1)
    CODE_GENDER: str | None = None
    NAME_CONTRACT_TYPE: str | None = None


REQUEST_SCHEMAS: dict[str, type[BaseModel]] = {
    "gmsc": GMSCApplication,
    "home_credit": HomeCreditApplication,
}


class ScoreResponse(BaseModel):
    probability_default: float = Field(ge=0, le=1, description="Predicted probability of default")
    score: int = Field(ge=300, le=850, description="Credit score, higher = lower risk")
    decision: Literal["approve", "decline"]
    threshold: float = Field(description="PD cut-off used for the decision")
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "model_not_loaded"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    dataset: str | None = None

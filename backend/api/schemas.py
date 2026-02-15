from pydantic import BaseModel
from typing import Dict


class PerformanceSchema(BaseModel):
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float


class PipelineOutputSchema(BaseModel):
    performance: PerformanceSchema
    latest_weights: Dict[str, float]
    stress_test: Dict

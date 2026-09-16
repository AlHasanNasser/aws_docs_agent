
from pydantic import BaseModel, Field
from datetime import datetime, timezone

class ChatRequest(BaseModel):
    """incoming request model for chat endpoint"""
    message:str = Field(
        ...,
        description="The message to send to the agent",
        max_length=10000,
        min_length=1
    ) 

    thread_id: str = Field(
        default = "default",
        description="The thread ID for the conversation. If not provided, a default thread will be used."
    )


class ChatResponse(BaseModel):
    """outgoing response model for chat endpoint"""
    response:str
    thread_id: str
    model_used: str
    cached: bool = False
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="The timestamp when the response was generated.")



class HealthResponse(BaseModel):
    """outgoing response model for health check endpoint"""
    status: str = "healthy"
    environment: str
    version: str = "1.0.0"
    checks: dict = {}

class MetricsResponse (BaseModel):
    """outgoing response model for metrics endpoint"""
    total_requests: int
    total_errors: int
    error_rate: str
    avg_latency_ms: float
    cache_hit_rate: str
    total_input_tokens: int  
    total_output_tokens: int


class ErrorResponse(BaseModel):
    """outgoing response model for error responses"""
    error: str
    details: str | None = None
    request_id: str | None = None

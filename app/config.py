
# Centeralized configuration
# USES pydantic BaseSettings to load environment variables and provide default values


from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    google_api_key: str
    primary_model: str = "gemini-3.1-flash-lite"
    fallback_model: str = "gemini-2.5-flash"
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "aws-rag"

    app_env : str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3

    model_config = {"env_file" : ".env", "extra" : "ignore" }

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache()
def get_settings() -> Settings:
    #create a cached instance of Settings to avoid reloading environment variables multiple times
    return Settings()

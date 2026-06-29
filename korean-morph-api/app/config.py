from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    openai_model: str = "gpt-4o"

    krdict_api_key: str
    krdict_base_url: str = "https://opendict.korean.go.kr/api"

    fetch_sense_detail: bool = True
    max_senses_per_word: int = 12


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

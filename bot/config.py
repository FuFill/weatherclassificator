from pydantic_settings import BaseSettings


class BotSettings(BaseSettings):
    """Bot configuration loaded from environment variables."""

    # Telegram
    bot_token: str = ""

    # ViT Classifier Service
    classifier_api_base_url: str = "http://localhost:8001"

    # LLM (Qwen)
    llm_api_base_url: str = ""
    llm_api_key: str = ""
    llm_api_model: str = "qwen3-coder-plus"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = BotSettings()

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENV: str
    PINECONE_INDEX_NAME: str
    PINECONE_DIMENSION: int = 1536
    PINECONE_METRIC: str = 'cosine'
    PINECONE_CLOUD: str = 'aws'
    PINECONE_REGION: str = 'us-east-1'

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()

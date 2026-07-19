from functools import lru_cache 
from pydantic_settings import BaseSettings,SettingsConfigDict


# lru_cache:least_recently_used cache store the limited number of item and discard the item which is recently least used 
# this improve performence by avoiding the repeated data acces and compute

### BaseSettings:subclass of base model for loading the setting from environment veriable, .env file

"""
SettingConfigDict : configuration setting is used inside the BaseSetting to control the how model
 load setting """
            
            
class Setting(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        
    )
    
    # Application
    app_name: str = "my_app"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "production"
    
    
    # Database 
    database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/myapp"
    database_pool_size: int = 20
    database_max_overflow: int = 10
    
    # Authentication
    secret_key: str
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # External Services
    
    redis_url: str = "redis://localhost:6379/0"
    smtp_host: str = "localhost"
    smtp_port: str = 587
    
    # CORS
    allowed_origins: list[str] = ["http://localhost:3000"]
    
    
    @property 
    def async_database_url(self) -> str:
        return self.database_url.replace(
            "postgresql://",
            "postgresql+asyncpg://"
        )
@lru_cache
def get_setting() -> Setting:
    return Setting

    """@lru-cache:The @lru_cache decorator ensures settings are loaded once and reused 
    — no repeated file reads or environment lookups on every request.
    """
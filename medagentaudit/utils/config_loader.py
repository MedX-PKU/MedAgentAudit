'''./MedagentAudit/utils/config_loader.py'''
import sys
from pydantic import BaseModel, Field
from pathlib import Path
import toml
from loguru import logger

class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    api_key: str = Field(..., description="API key for the LLM provider.")
    base_url: str = Field(..., description="Base URL for the LLM API.")
    model_name: str = Field(..., description="Model name to use.")
    timeout: int = Field(180, description="Request timeout in seconds.")
    stream: bool = Field(False, description="Whether to use streaming responses.")
    timeout: int = Field(180, description="Request timeout in seconds.")
    reasoning_effort: str = Field('medium', description="Level of reasoning effort: low, medium, high.")

class SystemConfig(BaseModel):
    max_retries: int = Field(3, description="Maximum number of retries for a failing task.")

class MedagentAudit(BaseModel):
    """The main configuration object, assembled dynamically from the config file."""
    active_llm_name: str
    llm: LLMProviderConfig
    system: SystemConfig

def get_config(config_path: str, active_llm: str) -> MedagentAudit:
    """
    Loads configuration from a TOML file, validates it, selects the active LLM's
    settings, and returns a unified object.
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'.")

    try:
        config_data = toml.load(config_path)

        if not active_llm:
            raise ValueError("'active_llm' parameter is required")

        active_llm_config_data = config_data.get("llm", {}).get(active_llm)
        if not active_llm_config_data:
            raise ValueError(f"Configuration for LLM '{active_llm}' not found under the '[llm]' section.")

        llm_provider_config = LLMProviderConfig(**active_llm_config_data)
        system_config = SystemConfig(**config_data.get("system", {}))

        config = MedagentAudit(
            active_llm_name=active_llm,
            llm=llm_provider_config,
            system=system_config,
        )
            
        logger.info(f"Configuration loaded successfully. Active LLM for reasoning: '{active_llm}'")
        return config

    except Exception as e:
        logger.error(f"Error parsing configuration file '{config_path}': {e}")
        raise ValueError(f"Error parsing configuration file '{config_path}': {e}") from e

def setup_logging(config: MedagentAudit):
    """Configures the Loguru logger based on the loaded configuration."""
    logger.remove()
    # Console logger
    logger.add(
        sys.stderr,
        level=config.logging.log_level.upper()
    )
    # File logger
    logger.add(
        config.logging.log_file,
        level=config.logging.log_level.upper(),
        rotation="10 MB",
        enqueue=True, # Make logging non-blocking
        backtrace=True,
        diagnose=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    logger.info("Logger configured.")
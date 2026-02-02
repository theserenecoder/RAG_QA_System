## Loggin configuration

import logging
import sys
from functools import lru_cache


def setup_logging(log_level: str = "INFO") ->None:
    """Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    
    formatter = logging.Formatter(
        fmt =  "[%(asctime)s] [%(levelname)s] [%(name)s] [%(message)s]",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    
    ## configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging,log_level.upper(),logging.INFO))
    
    ## remove existing handlers
    for handler in root_logger.handler[:]:
        root_logger.removeHandler()
        
    ## consol handler
    consol_handler = logging.StreamHandler(sys.stdout)
    consol_handler.setFormatter(formatter)
    
    root_logger.addHandler(consol_handler)
    
    
    ## reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    
    ## only for local implementations and not for productions
@lru_cache
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
    
class LoggerMixing:
    """mixing class to add logging capability to classes"""
    
    @property
    def logger(self)->logging.Logger:
        "Get logger for this class"
        return get_logger(self.__class__.__name__)
        
        
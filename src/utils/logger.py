"""logger.py

Gestisce il logging centralizzato del progetto.
Legge la configurazione da config.yaml e scrive messaggi su terminale e su file.

Funzioni:
- get_logger(name, config=None): restituisce un logger configurato pronto all'uso.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any, Union


# Configurazione di default per logging
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_DIR = Path("outputs/logs")


def get_logger(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Restituisce un logger configurato con StreamHandler e FileHandler.
    
    Args:
        name: Nome del logger (tipicamente __name__).
        config: Dizionario di configurazione con sezione 'logging' (opzionale).
                Struttura attesa:
                {
                    "logging": {
                        "level": "INFO",
                        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    },
                    "paths": {
                        "logs": "outputs/logs"
                    }
                }
        log_dir: Percorso della directory per i log. Sovrascrive config['paths']['logs'].
                 Se None, usa il valore da config o il default.
        
    Returns:
        Logger configurato con StreamHandler (console) e FileHandler (file).
    """
    logger = logging.getLogger(name)
    
    # Evita handler duplicati se il logger viene richiesto più volte
    if logger.hasHandlers():
        return logger
    
    # Estrai configurazione da config o usa default
    log_config = {}
    if config and "logging" in config:
        log_config = config["logging"]
    
    log_level_str = log_config.get("level", "INFO").upper()
    try:
        log_level = getattr(logging, log_level_str)
    except AttributeError:
        log_level = DEFAULT_LOG_LEVEL
    
    log_format_str = log_config.get("format", DEFAULT_LOG_FORMAT)
    formatter = logging.Formatter(log_format_str)
    
    logger.setLevel(log_level)
    
    # StreamHandler (console output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # FileHandler (file output)
    # Determina la directory dei log
    if log_dir is None:
        if config and "paths" in config and "logs" in config["paths"]:
            log_dir = Path(config["paths"]["logs"])
        else:
            log_dir = DEFAULT_LOG_DIR
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Nome del file di log basato sul nome del logger
    log_file = log_dir / f"{name.replace('.', '_')}.log"
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

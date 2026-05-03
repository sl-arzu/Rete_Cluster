"""config_loader.py

Carica e valida il file config.yaml.
Restituisce un dizionario Python con tutti i parametri della rete.

Funzioni:
- load_config(path): legge il YAML e restituisce le impostazioni.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Carica la configurazione dal file YAML.
    
    Args:
        config_path: Percorso al file config.yaml (str o Path).
        
    Returns:
        Dizionario con tutti i parametri di configurazione.
        
    Raises:
        FileNotFoundError: Se il file config non esiste.
        yaml.YAMLError: Se il file YAML è malformato.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML config {config_path}: {e}")
    
    if config is None:
        config = {}
    
    return config

"""factory.py

Factory pattern per creare neuroni dal config.yaml.
"""

from typing import Dict, Any

from src.neurons.base import BaseNeuron
from src.neurons.lif import LIFNeuron
from src.neurons.alif import ALIFNeuron


def create_neuron(
    n_neurons: int,
    neuron_type: str,
    config: Dict[str, Any],
) -> BaseNeuron:
    """
    Crea un neurone del tipo specificato usando i parametri dal config.
    
    Args:
        n_neurons: Numero di neuroni nel gruppo.
        neuron_type: Tipo di neurone ("LIF" o "ALIF").
        config: Dizionario di configurazione (sezione "neuron" da config.yaml).
    
    Returns:
        Istanza di BaseNeuron (LIFNeuron, ALIFNeuron, ecc.).
    
    Raises:
        ValueError: Se neuron_type non è riconosciuto.
        KeyError: Se i parametri richiesti mancano nel config.
    """
    neuron_type = neuron_type.upper()
    
    # Parametri comuni di rumore
    noise_enabled = config.get('noise', {}).get('enabled', False)
    noise_mu = config.get('noise', {}).get('mu', 0.0)
    noise_sigma = config.get('noise', {}).get('sigma', 0.1)
    
    if neuron_type == "LIF":
        # LIF: legge parametri da config["lif"]
        lif_config = config.get('lif', {})
        
        neuron = LIFNeuron(
            n_neurons=n_neurons,
            tau_mem=lif_config.get('tau_mem', 20.0),
            v_thresh=lif_config.get('v_thresh', 1.0),
            v_reset=lif_config.get('v_reset', 0.0),
            noise_enabled=noise_enabled,
            noise_mu=noise_mu,
            noise_sigma=noise_sigma,
        )
        return neuron
    
    elif neuron_type == "ALIF":
        # ALIF: legge parametri da config["alif"]
        alif_config = config.get('alif', {})
        
        neuron = ALIFNeuron(
            n_neurons=n_neurons,
            tau_mem=alif_config.get('tau_mem', 20.0),
            tau_adapt=alif_config.get('tau_adapt', 50.0),
            v_thresh=alif_config.get('v_thresh', 1.0),
            v_thresh_adapt_increment=alif_config.get('v_thresh_adapt_increment', 0.1),
            v_reset=alif_config.get('v_reset', 0.0),
            noise_enabled=noise_enabled,
            noise_mu=noise_mu,
            noise_sigma=noise_sigma,
        )
        return neuron
    
    else:
        raise ValueError(
            f"Neuron type '{neuron_type}' not recognized. "
            f"Supported types: 'LIF', 'ALIF'"
        )

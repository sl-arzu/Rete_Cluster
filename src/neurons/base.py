"""base.py

Interfaccia astratta (ABC) per i neuroni.
Ogni tipo di neurone (LIF, ALIF, ecc.) deve implementare questa interfaccia.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional


class BaseNeuron(nn.Module, ABC):
    """
    Interfaccia astratta per i neuroni spiking.
    
    Ogni neurone:
    - Riceve corrente di input e stato precedente
    - Genera spike (0 o 1) e stato aggiornato
    - Mantiene il proprio stato interno (membrana, soglia adattiva, ecc.)
    """
    
    def __init__(
        self,
        n_neurons: int,
        noise_enabled: bool = False,
        noise_mu: float = 0.0,
        noise_sigma: float = 0.1,
    ):
        """
        Inizializza il neurone.
        
        Args:
            n_neurons: Numero di neuroni nel gruppo.
            noise_enabled: Se True, applica rumore gaussiano alla corrente di input.
            noise_mu: Media del rumore gaussiano.
            noise_sigma: Deviazione standard del rumore gaussiano.
        """
        super().__init__()
        self.n_neurons = n_neurons
        self.noise_enabled = noise_enabled
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
    
    @abstractmethod
    def forward(
        self,
        input_current: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Esegue un step temporale del neurone.
        
        Args:
            input_current: Corrente di input, shape (batch_size, n_neurons).
            state: Dizionario con lo stato precedente (mem, adapt_thresh, ecc.).
                   Se None, usa init_state con batch_size appropriato.
        
        Returns:
            spike: Tensor di spike, shape (batch_size, n_neurons), valori 0 o 1.
            new_state: Dizionario con stato aggiornato.
        """
        raise NotImplementedError
    
    @abstractmethod
    def init_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Inizializza lo stato del neurone.
        
        Args:
            batch_size: Numero di campioni nel batch.
            device: Device (cpu o cuda) dove allocare i tensori.
        
        Returns:
            Dizionario con lo stato iniziale.
        """
        raise NotImplementedError
    
    def _apply_noise(self, input_current: torch.Tensor) -> torch.Tensor:
        """
        Applica rumore gaussiano alla corrente di input.
        
        Args:
            input_current: Corrente di input, shape (batch_size, n_neurons).
        
        Returns:
            input_current con rumore aggiunto (se noise_enabled=True).
        """
        if self.noise_enabled:
            noise = torch.randn_like(input_current) * self.noise_sigma + self.noise_mu
            return input_current + noise
        return input_current

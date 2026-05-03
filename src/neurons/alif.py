"""alif.py

ALIF (Adaptive Leaky Integrate-and-Fire) Neuron.
Estende LIF con soglia adattiva che aumenta dopo ogni spike.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

from src.neurons.base import BaseNeuron


class ALIFNeuron(BaseNeuron):
    """
    ALIF neuron con soglia adattiva.
    
    Dinamica:
    - mem[t] = beta * mem[t-1] + (1-beta) * input[t]
    - threshold[t] = v_thresh_base + adapt_thresh[t]
    - adapt_thresh[t] = beta_adapt * adapt_thresh[t-1] + v_thresh_adapt_increment * spike[t-1]
    - Se mem[t] > threshold[t]: spike=1, mem[t]=reset, adapt_thresh[t] += increment
    
    Parametri:
    - tau_mem: Costante di tempo della membrana
    - tau_adapt: Costante di tempo del decadimento della soglia adattiva
    - v_thresh: Soglia di base
    - v_thresh_adapt_increment: Quanto aumenta la soglia dopo ogni spike
    - v_reset: Potenziale di reset
    """
    
    def __init__(
        self,
        n_neurons: int,
        tau_mem: float = 20.0,
        tau_adapt: float = 50.0,
        v_thresh: float = 1.0,
        v_thresh_adapt_increment: float = 0.1,
        v_reset: float = 0.0,
        noise_enabled: bool = False,
        noise_mu: float = 0.0,
        noise_sigma: float = 0.1,
    ):
        """
        Inizializza l'ALIF neuron.
        
        Args:
            n_neurons: Numero di neuroni.
            tau_mem: Costante di tempo della membrana.
            tau_adapt: Costante di tempo del decadimento soglia adattiva.
            v_thresh: Soglia di base.
            v_thresh_adapt_increment: Incremento di soglia per spike.
            v_reset: Potenziale di reset.
            noise_enabled: Se True, applica rumore gaussiano.
            noise_mu: Media del rumore.
            noise_sigma: Std dev del rumore.
        """
        super().__init__(
            n_neurons=n_neurons,
            noise_enabled=noise_enabled,
            noise_mu=noise_mu,
            noise_sigma=noise_sigma,
        )
        
        self.tau_mem = tau_mem
        self.tau_adapt = tau_adapt
        self.v_thresh = v_thresh
        self.v_thresh_adapt_increment = v_thresh_adapt_increment
        self.v_reset = v_reset
        
        # Coefficienti di decadimento
        self.beta_mem = 1.0 - 1.0 / tau_mem
        self.beta_adapt = 1.0 - 1.0 / tau_adapt
    
    def forward(
        self,
        input_current: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass per un singolo step temporale.
        
        Args:
            input_current: Shape (batch_size, n_neurons).
            state: Dict con 'mem' e 'adapt_thresh' precedenti.
        
        Returns:
            spike: Shape (batch_size, n_neurons), valori 0 o 1.
            new_state: Dict con 'mem' e 'adapt_thresh' aggiornati.
        """
        batch_size = input_current.shape[0]
        device = input_current.device
        
        # Applica rumore
        input_current = self._apply_noise(input_current)
        
        # Estrai stato precedente
        if state is None:
            mem = torch.zeros((batch_size, self.n_neurons), device=device, dtype=input_current.dtype)
            adapt_thresh = torch.zeros((batch_size, self.n_neurons), device=device, dtype=input_current.dtype)
        else:
            mem = state.get('mem', torch.zeros((batch_size, self.n_neurons), device=device, dtype=input_current.dtype))
            adapt_thresh = state.get('adapt_thresh', torch.zeros((batch_size, self.n_neurons), device=device, dtype=input_current.dtype))
        
        # Integra la membrana: mem[t] = beta * mem[t-1] + (1-beta) * input[t]
        mem_new = self.beta_mem * mem + (1.0 - self.beta_mem) * input_current
        
        # Calcola la soglia adattiva: threshold[t] = v_thresh + adapt_thresh[t]
        threshold = self.v_thresh + adapt_thresh
        
        # Genera spike: mem > threshold → spike=1
        spike = (mem_new > threshold).float()
        
        # Reset della membrana dopo spike
        mem_new = mem_new * (1.0 - spike) + self.v_reset * spike
        
        # Aggiorna la soglia adattiva:
        # adapt_thresh[t] = beta_adapt * adapt_thresh[t-1] + v_thresh_adapt_increment * spike[t]
        adapt_thresh_new = self.beta_adapt * adapt_thresh + self.v_thresh_adapt_increment * spike
        
        # Crea il nuovo stato
        new_state = {
            'mem': mem_new,
            'adapt_thresh': adapt_thresh_new,
        }
        
        return spike, new_state
    
    def init_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Inizializza lo stato del neurone.
        
        Args:
            batch_size: Numero di campioni.
            device: Device dove allocare.
        
        Returns:
            Dict con 'mem' e 'adapt_thresh' iniziali.
        """
        state = {
            'mem': torch.zeros((batch_size, self.n_neurons), device=device, dtype=torch.float32),
            'adapt_thresh': torch.zeros((batch_size, self.n_neurons), device=device, dtype=torch.float32),
        }
        return state
    
    def extra_repr(self) -> str:
        """Rappresentazione stringa del modulo."""
        return (
            f"n_neurons={self.n_neurons}, "
            f"tau_mem={self.tau_mem}, "
            f"tau_adapt={self.tau_adapt}, "
            f"v_thresh={self.v_thresh}, "
            f"v_thresh_adapt_increment={self.v_thresh_adapt_increment}, "
            f"noise_enabled={self.noise_enabled}"
        )

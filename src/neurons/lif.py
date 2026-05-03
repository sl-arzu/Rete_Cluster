"""lif.py

LIF (Leaky Integrate-and-Fire) Neuron usando snnTorch.
Wrappa snn.Leaky per aggiungere rumore dinamico e facilità di uso.
"""

import torch
import torch.nn as nn
import snntorch as snn
from typing import Tuple, Dict, Optional

from src.neurons.base import BaseNeuron


class LIFNeuron(BaseNeuron):
    """
    LIF neuron implementato con snnTorch.
    
    Dinamica:
    - mem[t] = beta * mem[t-1] + (1-beta) * input[t]
    - Se mem[t] > threshold: spike=1, mem[t]=reset
    - Altrimenti: spike=0
    
    Parametri:
    - tau_mem: Costante di tempo della membrana (tau_mem = 1/beta, quindi beta = 1 - exp(-1/tau_mem))
    - v_thresh: Soglia di spike
    - v_reset: Potenziale di reset dopo spike
    """
    
    def __init__(
        self,
        n_neurons: int,
        tau_mem: float = 20.0,
        v_thresh: float = 1.0,
        v_reset: float = 0.0,
        noise_enabled: bool = False,
        noise_mu: float = 0.0,
        noise_sigma: float = 0.1,
    ):
        """
        Inizializza il LIF neuron.
        
        Args:
            n_neurons: Numero di neuroni.
            tau_mem: Costante di tempo della membrana.
            v_thresh: Soglia di spike.
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
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        
        # Crea il neurone snnTorch
        # beta = 1 - 1/tau_mem è il decay rate
        self.beta = 1.0 - 1.0 / tau_mem
        
        # Creiamo il Leaky neuron di snnTorch
        # Nota: snn.Leaky richiede input shape (time, batch, neurons) o (batch, neurons)
        # Noi usiamo (batch, neurons) per singoli step temporali
        self.lif_cell = snn.Leaky(
            beta=self.beta,
            threshold=v_thresh,
            reset_mechanism="zero" if v_reset == 0.0 else "subtract",
            init_hidden=False,
        )
        
        # Se reset è non-zero, dobbiamo gestirlo manualmente
        self.v_reset_manual = v_reset if v_reset != 0.0 else None
    
    def forward(
        self,
        input_current: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass per un singolo step temporale.
        
        Args:
            input_current: Shape (batch_size, n_neurons).
            state: Dict con 'mem' precedente.
        
        Returns:
            spike: Shape (batch_size, n_neurons), valori 0 o 1.
            new_state: Dict con 'mem' aggiornato.
        """
        # Applica rumore
        input_current = self._apply_noise(input_current)
        
        # Estrai la membrana precedente
        if state is None or 'mem' not in state:
            mem = torch.zeros(input_current.shape, device=input_current.device, dtype=input_current.dtype)
        else:
            mem = state['mem']
        
        # Forward pass attraverso il LIF cell
        spike, mem_new = self.lif_cell(input_current, mem)
        
        # Gestisci reset manuale se necessario
        if self.v_reset_manual is not None:
            # Dopo spike, la membrana dovrebbe andare a v_reset
            # snn.Leaky con reset="zero" mette a 0, noi vogliamo v_reset
            mem_new = mem_new * (1.0 - spike) + self.v_reset_manual * spike
        
        # Crea il nuovo stato
        new_state = {'mem': mem_new}
        
        return spike, new_state
    
    def init_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Inizializza lo stato del neurone (membrana a riposo).
        
        Args:
            batch_size: Numero di campioni.
            device: Device dove allocare.
        
        Returns:
            Dict con 'mem' iniziale (tutti zeri).
        """
        mem = torch.zeros(
            (batch_size, self.n_neurons),
            device=device,
            dtype=torch.float32,
        )
        return {'mem': mem}
    
    def extra_repr(self) -> str:
        """Rappresentazione stringa del modulo."""
        return (
            f"n_neurons={self.n_neurons}, "
            f"tau_mem={self.tau_mem}, "
            f"v_thresh={self.v_thresh}, "
            f"v_reset={self.v_reset}, "
            f"noise_enabled={self.noise_enabled}"
        )

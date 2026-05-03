"""population_encoder.py

Population-based rate encoding per convertire segnali analogici in spike trains.

Ogni feature input genera una sottobanda di 'population_size' neuroni encoder.
Ogni neurone emette spike con frequenza (rate Poisson) proporzionale al segnale ponderato.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

from src.utils.logger import get_logger


class PopulationEncoder(nn.Module):
    """
    Encoder che converte segnali analogici in spike trains usando population coding.
    
    Flusso:
    1. Input (batch_size, n_features) viene ponderato da una matrice (n_features, population_size)
    2. Rumore gaussiano opzionale viene aggiunto
    3. I rate (frequenze di spike) vengono clampati a [0, max_rate]
    4. Spike train viene generato da distribuzione di Poisson per nb_steps
    5. Output: (batch_size, nb_steps, n_features * population_size)
    """
    
    def __init__(
        self,
        n_features: int,
        population_size: int,
        nb_steps: int,
        weight_mu: float = 0.0,
        weight_sigma: float = 0.1,
        noise_enabled: bool = True,
        noise_mu: float = 0.0,
        noise_sigma: float = 0.1,
        max_rate: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Inizializza il PopulationEncoder.
        
        Args:
            n_features: Numero di feature di input (es. 400).
            population_size: Numero di neuroni per feature (es. 10).
            nb_steps: Numero di step temporali per l'encoding (es. 100).
            weight_mu: Media della distribuzione gaussiana per i pesi.
            weight_sigma: Deviazione standard dei pesi.
            noise_enabled: Se True, applica rumore gaussiano ad ogni forward.
            noise_mu: Media della distribuzione del rumore.
            noise_sigma: Deviazione standard del rumore.
            max_rate: Rate massimo per la generazione Poisson (clamping).
            config: Dizionario di configurazione per il logger.
        """
        super().__init__()
        
        self.n_features = n_features
        self.population_size = population_size
        self.nb_steps = nb_steps
        self.noise_enabled = noise_enabled
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.max_rate = max_rate
        
        # Logger
        self.logger = get_logger(__name__, config=config or {})
        
        # Pesi sinaptici: (n_features, population_size)
        # Inizializzazione gaussiana: N(weight_mu, weight_sigma^2)
        self.weights = nn.Parameter(
            torch.randn(n_features, population_size) * weight_sigma + weight_mu
        )
        
        # Flag per loggare una volta sola al primo forward
        self._first_forward = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Codifica segnali analogici in spike trains usando population coding.
        
        Args:
            x: Tensor di shape (batch_size, n_features) con valori analogici.
            
        Returns:
            Tensor di shape (batch_size, nb_steps, n_features * population_size)
            contenente spike trains (0 o 1) generati da Poisson.
        """
        batch_size = x.shape[0]
        
        # Log al primo forward
        if self._first_forward:
            output_shape = (batch_size, self.nb_steps, self.n_features * self.population_size)
            self.logger.info(
                f"PopulationEncoder: n_features={self.n_features}, "
                f"population_size={self.population_size}, output_shape={output_shape}"
            )
            self._first_forward = False
        
        # 1. Espandi input: (batch_size, n_features) → (batch_size, n_features, 1)
        x_expanded = x.unsqueeze(-1)  # shape: (batch_size, n_features, 1)
        
        # 2. Applica pesi: (batch_size, n_features, 1) * (n_features, population_size)
        #    Broadcasting: (B, n_features, 1) * (1, n_features, population_size) 
        #    → (B, n_features, population_size)
        weights_unsqueezed = self.weights.unsqueeze(0)  # (1, n_features, population_size)
        weighted = x_expanded * weights_unsqueezed  # (B, n_features, population_size)
        
        # 3. Aggiungi rumore gaussiano se abilitato
        if self.noise_enabled:
            noise = torch.randn_like(weighted) * self.noise_sigma + self.noise_mu
            weighted = weighted + noise
        
        # 4. Reshape per appiattire le dimensioni (n_features, population_size) → (n_features * population_size)
        #    (B, n_features, population_size) → (B, n_features * population_size)
        rates = weighted.view(batch_size, -1)
        
        # 5. Clamp i rate a valori positivi (per Poisson) e sotto max_rate
        rates = torch.clamp(rates, min=0.0, max=self.max_rate)
        
        # 6. Genera spike trains Poisson per nb_steps
        #    Espandi per il numero di step: (B, n_features*pop_size) → (B, nb_steps, n_features*pop_size)
        rates_expanded = rates.unsqueeze(1).expand(-1, self.nb_steps, -1)
        
        # Campiona da Poisson(rate) per ogni elemento
        spike_train = torch.poisson(rates_expanded)
        
        # 7. Binarizza i spike (convert poisson counts → binary spikes)
        spike_train = (spike_train > 0).float()
        
        return spike_train
    
    def extra_repr(self) -> str:
        """Rappresentazione stringa del modulo."""
        return (
            f"n_features={self.n_features}, "
            f"population_size={self.population_size}, "
            f"nb_steps={self.nb_steps}, "
            f"noise_enabled={self.noise_enabled}, "
            f"max_rate={self.max_rate}"
        )

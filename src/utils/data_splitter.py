"""data_splitter.py

Divide dataset grezzi (X, y) in Train/Val/Test con stratificazione.
Crea DataLoader PyTorch pronti per l'addestramento.

Funzioni principali:
- DataSplitter: Classe per lo split stratificato e creazione di DataLoader.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Union, overload
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger


class TensorDataset(Dataset):
    """Dataset semplice da array NumPy per PyTorch."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Array NumPy di features, shape (n_samples, ...).
            y: Array NumPy di labels, shape (n_samples,).
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DataSplitter:
    """
    Divide dataset grezzi in Train/Val/Test con stratificazione.
    
    Modalità supportate:
    - "train_test": 2 split (Train/Test)
    - "train_val_test": 3 split (Train/Val/Test)
    """
    
    # Type annotations per le variabili instance
    X: np.ndarray
    y: np.ndarray
    mode: str
    train_ratio: float
    val_ratio: Optional[float]  # None se mode="train_test"
    test_ratio: float
    stratify: bool
    random_seed: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    config: Dict
    logger: logging.Logger
    
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_val: Optional[np.ndarray]
    y_val: Optional[np.ndarray]
    
    train_loader: DataLoader
    test_loader: DataLoader
    val_loader: Optional[DataLoader]
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mode: str = "train_val_test",
        train_ratio: float = 0.70,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        stratify: bool = True,
        random_seed: int = 42,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        config: Optional[Dict] = None,
    ):
        """
        Inizializza il DataSplitter.
        
        Args:
            X: Array NumPy di shape (n_samples, ...).
            y: Array NumPy di shape (n_samples,) con labels intere.
            mode: "train_test" o "train_val_test".
            train_ratio: Proporzione per training (0.0-1.0).
            val_ratio: Proporzione per validation. 
                       - Obbligatorio se mode="train_val_test" (default 0.15).
                       - Ignorato se mode="train_test".
            test_ratio: Proporzione per test.
                       - Default 0.3 se mode="train_test", 0.15 se mode="train_val_test".
            stratify: Se True, mantiene distribuzione classi in ogni split.
            random_seed: Seed per la riproducibilità.
            batch_size: Dimensione del batch per DataLoader.
            num_workers: Numero di worker per DataLoader.
            pin_memory: Se True, pinma memoria GPU (utile se device="cuda").
            config: Dizionario di configurazione (per logger).
        """
        # Imposta i default corretti in base alla modalità
        if mode == "train_test":
            if test_ratio is None:
                test_ratio = 0.3
            if val_ratio is not None:
                import warnings
                warnings.warn(
                    "mode='train_test' ignora val_ratio. Usare mode='train_val_test' per la validazione.",
                    UserWarning
                )
                val_ratio = None
        else:  # train_val_test
            if val_ratio is None:
                val_ratio = 0.15
            if test_ratio is None:
                test_ratio = 0.15
        
        self.X = X
        self.y = y
        self.mode = mode
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.stratify = stratify
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.config = config or {}
        
        # Logger
        self.logger = get_logger(__name__, config=self.config)
        
        # Validazione
        self._validate_inputs()
        
        # Split e creazione DataLoader
        self._split_data()
        self._create_dataloaders()
        self._log_summary()
    
    def _validate_inputs(self):
        """Valida gli input."""
        if len(self.X) != len(self.y):
            raise ValueError(f"X e y devono avere lo stesso numero di campioni. Got {len(self.X)} vs {len(self.y)}")
        
        if self.mode not in ["train_test", "train_val_test"]:
            raise ValueError(f"mode deve essere 'train_test' o 'train_val_test'. Got {self.mode}")
        
        if self.mode == "train_test":
            # Verifica che train_ratio e test_ratio sommino a 1.0
            if not np.isclose(self.train_ratio + self.test_ratio, 1.0):
                raise ValueError(
                    f"train_ratio + test_ratio deve essere 1.0. "
                    f"Got {self.train_ratio} + {self.test_ratio} = {self.train_ratio + self.test_ratio}"
                )
        else:  # train_val_test
            # Verifica che tutti e tre i ratio siano forniti e sommino a 1.0
            if not np.isclose(self.train_ratio + self.val_ratio + self.test_ratio, 1.0):
                raise ValueError(
                    f"train_ratio + val_ratio + test_ratio deve essere 1.0. "
                    f"Got {self.train_ratio} + {self.val_ratio} + {self.test_ratio} = "
                    f"{self.train_ratio + self.val_ratio + self.test_ratio}"
                )
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size deve essere > 0. Got {self.batch_size}")
    
    def _split_data(self):
        """Esegue lo split stratificato."""
        stratify_labels = self.y if self.stratify else None
        
        if self.mode == "train_test":
            # Split in Train e Test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y,
                train_size=self.train_ratio,
                test_size=self.test_ratio,
                stratify=stratify_labels,
                random_state=self.random_seed,
            )
            self.X_val: Optional[np.ndarray] = None
            self.y_val: Optional[np.ndarray] = None
        
        else:  # train_val_test
            # Prima split: Train+Val vs Test
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                self.X, self.y,
                test_size=self.test_ratio,
                stratify=stratify_labels,
                random_state=self.random_seed,
            )
            
            # Secondo split: Train vs Val
            # Ricalcola le proporzioni per il subset Train+Val
            assert self.val_ratio is not None, "val_ratio must be set for train_val_test mode"
            val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio_adjusted,
                stratify=y_temp if self.stratify else None,
                random_state=self.random_seed,
            )
    
    def _create_dataloaders(self):
        """Crea DataLoader PyTorch."""
        train_dataset = TensorDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        test_dataset = TensorDataset(self.X_test, self.y_test)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        if self.mode == "train_val_test":
            assert self.X_val is not None and self.y_val is not None, \
                "X_val and y_val must be initialized for train_val_test mode"
            val_dataset = TensorDataset(self.X_val, self.y_val)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else:
            self.val_loader = None
    
    def _log_summary(self):
        """Registra il sommario dello split."""
        n_train = len(self.X_train)
        n_test = len(self.X_test)
        n_val = len(self.X_val) if self.X_val is not None else 0
        
        # Classi uniche e loro conteggi
        unique_classes = np.unique(self.y)
        classes_str = ", ".join([str(c) for c in unique_classes])
        
        if self.mode == "train_val_test":
            self.logger.info(
                f"Split completato — Train: {n_train}, Val: {n_val}, Test: {n_test}, "
                f"Classi: [{classes_str}]"
            )
        else:
            self.logger.info(
                f"Split completato — Train: {n_train}, Test: {n_test}, "
                f"Classi: [{classes_str}]"
            )
        
        # Log dettagliato della distribuzione classi
        for split_name, labels in [
            ("Train", self.y_train),
            ("Val", self.y_val if self.y_val is not None else None),
            ("Test", self.y_test),
        ]:
            if labels is None:
                continue
            unique, counts = np.unique(labels, return_counts=True)
            dist_str = ", ".join([f"class_{c}={cnt}" for c, cnt in zip(unique, counts)])
            self.logger.debug(f"{split_name} class distribution: {dist_str}")
    
    @overload
    def get_loaders(self: "DataSplitter") -> Tuple[DataLoader, DataLoader]:
        """Overload per mode='train_test'."""
        ...
    
    @overload
    def get_loaders(self: "DataSplitter") -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Overload per mode='train_val_test'."""
        ...
    
    def get_loaders(self) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Restituisce i DataLoader.
        
        Returns:
            - Se mode="train_test": (train_loader, test_loader)
            - Se mode="train_val_test": (train_loader, val_loader, test_loader)
        """
        if self.mode == "train_test":
            return self.train_loader, self.test_loader
        else:
            assert self.val_loader is not None, "val_loader must be initialized for train_val_test mode"
            return self.train_loader, self.val_loader, self.test_loader
    
    def get_splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Restituisce i dati grezzi (NumPy arrays) per ogni split.
        
        Returns:
            Dizionario con chiavi "train", "val" (opzionale), "test".
            Valori: (X, y) per ciascuno split.
        """
        splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "train": (self.X_train, self.y_train),
            "test": (self.X_test, self.y_test),
        }
        if self.mode == "train_val_test":
            assert self.X_val is not None and self.y_val is not None, \
                "X_val and y_val must be initialized for train_val_test mode"
            splits["val"] = (self.X_val, self.y_val)
        return splits

import sys
from pathlib import Path

# Aggiungi la root directory al path per permettere import di 'src'
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
from src.utils.data_splitter import DataSplitter

# 1. Carica il tuo dataset NPZ (sostituisci col percorso reale)
data = np.load(root_dir / "data/iron_stress/Iron_Stress.npz")
X = data["X"]  # shape (2016, 400)
y = data["y"]  # shape (2016,)

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Classi uniche: {np.unique(y)}")
print(f"Distribuzione: {dict(zip(*np.unique(y, return_counts=True)))}")

# 2. Test Modalità B — Train/Val/Test
splitter = DataSplitter(
    X, y,
    mode="train_val_test",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify=True,
    random_seed=42,
    batch_size=32,
)

train_loader, val_loader, test_loader = splitter.get_loaders()

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# 3. Verifica stratificazione: prendi un batch e conta le classi
for batch_X, batch_y in train_loader:
    print(f"\nPrimo batch train — shape: {batch_X.shape}")
    print(f"Classi nel batch: {dict(zip(*np.unique(batch_y.numpy(), return_counts=True)))}")
    break

# 4. Verifica splits grezzi
splits = splitter.get_splits()
for name, (X_split, y_split) in splits.items():
    print(f"{name}: X={X_split.shape}, classi={dict(zip(*np.unique(y_split, return_counts=True)))}")

print("\n✅ Test completato con successo!")
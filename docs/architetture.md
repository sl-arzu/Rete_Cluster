# === ARCHITECTURE.MD ===
architecture_md = """# Architecture Specification — RSNN Modulare per Classificazione Temporale

> **Progetto:** Tomato Stress / Bioimpedance (generalizzabile a dataset NPZ strutturati)  
> **Framework:** PyTorch  
> **Hardware dev:** MacBook M1 (venv locale)  
> **Hardware training:** server GPU remoto `snnmonster` via SSH  
> **Versione SPEC:** v0.1.0-spec  
> **Branch:** `feat/spec-architettura`

---

## 1. Input Encoding — Population Coding

Ogni segnale scalare in input alimenta una **popolazione** di `N` neuroni encoder. Il layer di input trasforma i segnali grezzi in spike trains compatibili con l'hidden layer.

### 1.1. Dimensioni e shape garantita
- Parametro configurabile: `population_size` (default suggerito: 10).
- Per ogni feature temporale si crea una popolazione di `population_size` neuroni.
- **Shape di output obbligatoria per tutti gli encoder:**
  ```
  (n_samples, nb_steps, n_features × population_size)
  ```
- Questa forma è invariante rispetto al tipo di encoder scelto (Poisson, rate-based, etc.), garantendo compatibilità con il layer successivo.

### 1.2. Pesi sinaptici iniziali
- I neuroni della stessa popolazione ricevono lo stesso segnale scalare in input.
- I pesi sinaptici sono inizializzati con una **gaussiana** configurabile:
  - `weight_mu` (μ): media della distribuzione
  - `weight_sigma` (σ): deviazione standard
- Ogni neurone nella popolazione ha un peso iniziale indipendente campionato da questa distribuzione.

### 1.3. Rumore additivo sugli encoder (diversificazione popolazione)
- **Parametri configurabili:**
  - `encoder_noise_enabled`: booleano, attiva/disattiva rumore
  - `encoder_noise_mu`: media del rumore gaussiano (default: 0.0)
  - `encoder_noise_sigma`: deviazione standard del rumore (default: 0.1)
- Il rumore è applicato **dinamicamente durante la forward pass** sul segnale (o sulla frequenza/rate) **prima** della conversione in spike.
- Ogni neurone della popolazione riceve rumore **indipendente** dagli altri.
- **Effetto:** anche se i neuroni hidden dello stesso cluster ricevono connessioni dalla stessa popolazione, il segnale trasmesso è leggermente diverso per ogni percorso, evitando comportamenti sincronizzati e garantendo diversità di attivazione.
- Encoder di tipo rate-based (Poisson/Hz) implementano questo rumore come perturbazione del rate di emissione.

---

## 2. Hidden Layer — Struttura a Cluster

L'hidden layer è il cuore computazionale della RSNN. È partizionato in cluster con regole di connettività rigide ma configurabili.

### 2.1. Partizione in cluster
- L'hidden layer è suddiviso in cluster, ciascuno associato a una specifica popolazione di input.
- Ogni neurone hidden appartiene a **uno e un solo cluster**, oppure è un **neurone orfano** (nessun cluster assegnato).
- Il mapping cluster ↔ popolazione è configurabile in fase di costruzione della rete.

### 2.2. Connettività input → hidden
- **Regola fondamentale:** ogni neurone hidden riceve **al massimo 1 connessione** dalla sua popolazione di riferimento.
- La connessione è con **un singolo neurone** della popolazione (non con tutta la popolazione).
- **Distribuzione non equa:** la percentuale di neuroni hidden collegati a ciascun neurone della popolazione è configurabile.
  - Esempio: 30% degli hidden collegati al neurone pop 1, 50% al neurone pop 2, ecc.
- Parametro chiave: `input_conn_distribution` (mappa percentuali).
- Parametro di controllo: `max_input_conn_per_hidden = 1` (vincolo architetturale).

### 2.3. Neuroni orfani
- Parametro configurabile: `orphan_ratio` (es. `0.10` = 10% degli hidden).
- Gli orfani **non ricevono alcuna connessione diretta** dagli input.
- Ricevono esclusivamente connessioni ricorrenti da altri neuroni hidden.
- Sono utili per memorizzare stati temporali puri senza bias da input diretti.

### 2.4. Connessioni ricorrenti hidden → hidden
- Le connessioni ricorrenti sono divise in due tipologie con limiti separati:
  1. **Intra-cluster:** tra neuroni appartenenti allo stesso cluster.
  2. **Inter-cluster:** tra neuroni di cluster diversi.
- Parametri configurabili:
  - `max_intra_cluster_conn`: numero massimo di connessioni ricorrenti intra-cluster per neurone.
  - `max_inter_cluster_conn`: numero massimo di connessioni ricorrenti inter-cluster per neurone.
- Le connessioni sono sparse e inizializzate secondo una distribuzione configurabile.

---

## 3. Output Layer

L'output layer produce le predizioni di classe a partire dallo stato spiking dell'hidden layer.

### 3.1. Numero di neuroni di output
- **Determinato automaticamente** dal numero di etichette diverse presenti nel dataset NPZ fornito.
- Se il dataset contiene 3 classi, l'output layer avrà 3 neuroni.
- Il conteggio delle classi uniche viene effettuato in fase di caricamento del dataset.

### 3.2. Connettività hidden → output
- Parametro configurabile: `max_outgoing_conn_per_output` (numero massimo di connessioni in ingresso che ciascun neurone di output può ricevere).
- Parametro configurabile: `hidden_to_output_ratio` (percentuale di neuroni hidden che possono essere collegati ai neuroni di output, es. `0.50` = 50%).
- La selezione dei neuroni hidden da collegare può essere random o basata su criterio (es. attività media).

---

## 4. Modularità Neuroni — Interfaccia Comune

Il sistema deve supportare diversi modelli di neuroni senza modificare la logica dei layer.

### 4.1. Tipi supportati
- **LIF** (Leaky Integrate-and-Fire)
- **ALIF** (Adaptive LIF)
- Altri tipi futuri estendibili tramite la stessa interfaccia.

### 4.2. Interfaccia
- Ogni neurone implementa una classe con metodi standardizzati:
  - `forward(spike_input, current_state) → (spike_output, next_state)`
- Il tipo di neurone è selezionabile via configurazione (`neuron.type` nel config YAML).
- Parametri di rumore gaussiano a livello neurono LIF/ALIF rimangono disponibili e indipendenti dal rumore encoder.

---

## 5. Training — Algoritmi Intercambiabili

Il sistema deve supportare più algoritmi di training senza modificare la definizione della rete.

### 5.1. Algoritmi supportati
- **BPTT** (Back-Propagation Through Time) — baseline standard con PyTorch `autograd`.
- **e-prop** (eligible propagation) — approccio bio-plausibile per SNN.
- Altri algoritmi configurabili in futuro tramite interfaccia comune.

### 5.2. Interfaccia training
- Classe base `Trainer` con metodi:
  - `train_epoch(model, dataloader) → metrics`
  - `validate(model, dataloader) → metrics`
- Implementazioni concrete: `BPTTTrainer`, `EPropTrainer`.
- Scambio trainer via configurazione (`training.algorithm`).

### 5.3. Hardware di training
- Training eseguito su server GPU remoto `snnmonster` via SSH.
- Dev locale su MacBook M1 con supporto MPS o CPU.

---

## 6. Inferenza — Script Separato

L'inferenza è decouplata dal training per garantire riproducibilità e analisi post-hoc.

### 6.1. Script di inference
- Script indipendente: `scripts/inference.py` (o equivalente).
- Carica modello e checkpoint senza aggiornamento pesi.
- Opera sul **test set** (mai visto durante training/validation).

### 6.2. Output prodotti
1. **Predizioni e metriche:**
   - Accuracy
   - F1-macro (entrambi richiesti come metriche obiettivo)
2. **Attività spike-by-spike:**
   - Array NumPy con spike raster (shape: `(n_samples, nb_steps, n_neurons_layer)`)
   - Pronti per visualizzazione (raster plots, heatmaps)
3. **Log strutturato:** salvataggio in `outputs/plots/` e `outputs/logs/`.

---

## 7. Gestione Dataset — DataSplitter

Il caricamento e lo split dei dati sono centralizzati in una classe dedicata.

### 7.1. Formato dataset atteso
- File NPZ con chiavi standard (es. `X`, `y`).
- Shape attesa: `(n_samples, nb_steps, n_features)` oppure `(n_samples, n_features)` con successiva espansione temporale.
- Per il progetto Tomato Stress:
  - 2016 campioni totali
  - 3 classi bilanciate: 0=Control (672), 1=Early Stress (672), 2=Late Stress (672)
  - 3 piante: P0, P1, P3
  - Frequenze rilevanti: Water Stress → indici 0–10, Iron Stress → indici 190–199

### 7.2. Split configurabile
La classe `DataSplitter` deve supportare due modalità controllate da parametri passabili in input:

| Modalità | Split | Uso |
|----------|-------|-----|
| A — Train/Test | Train / Test | Esempio: 70% / 30% |
| B — Train/Val/Test | Train / Val / Test | Esempio: 70% / 15% / 15% |

- Le proporzioni sono **parametrizzabili**, non hardcoded.
- Parametri consigliati nel config:
  - `split_mode`: `"train_test"` o `"train_val_test"`
  - `train_ratio`, `val_ratio`, `test_ratio`

### 7.3. Stratified split
- Lo split è **stratificato** per bilanciamento delle classi in ogni sotto-insieme.
- Parametro `stratify: true/false` nel config.
- Implementazione consigliata tramite `scikit-learn` (`StratifiedShuffleSplit` o `train_test_split` con `stratify=y`).

### 7.4. Interfaccia DataSplitter
```python
splitter = DataSplitter(
    X, y,
    mode="train_val_test",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify=True,
    random_seed=42
)
train_loader, val_loader, test_loader = splitter.get_loaders(batch_size=32)
```

---

## 8. Configurazione Centralizzata (YAML)

Tutti i parametri sopra descritti sono esposti tramite un file `config.yaml` unico.

### 8.1. Schema ad alta struttura
```yaml
dataset:
  path: "data/dataset.npz"
  split_mode: "train_val_test"
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  stratify: true

input_encoder:
  type: "rate_poisson"
  population_size: 10
  weight_mu: 0.0
  weight_sigma: 0.1
  noise:
    enabled: true
    mu: 0.0
    sigma: 0.1

hidden_layer:
  n_neurons: 256
  orphan_ratio: 0.10
  input_conn_distribution: { ... }
  recurrence:
    max_intra_cluster: 5
    max_inter_cluster: 3

output_layer:
  max_incoming_connections: 10
  hidden_to_output_ratio: 0.50

neuron:
  type: "LIF"

training:
  algorithm: "BPTT"
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

---

## 9. Decisioni Architetturali Chiave (ADR)

| ID | Decisione | Stato |
|---|---|---|
| ADR-001 | Connessioni ricorrenti intra + inter cluster con limiti separati | Accettata |
| ADR-002 | Neuroni orfani (`orphan_ratio` configurabile) | Accettata |
| ADR-003 | Rumore gaussiano su encoder per diversificazione popolazione; LIF/ALIF mantengono rumore proprio indipendente | Accettata (rev. 2026-04-29) |
| ADR-004 | Shape encoder `(n_samples, nb_steps, n_features × population_size)` invariante | Accettata |
| ADR-005 | Numero neuroni output dinamico da classi uniche NPZ | Accettata |
| ADR-006 | Training BPTT baseline, e-prop come estensione modulare | Accettata |

---
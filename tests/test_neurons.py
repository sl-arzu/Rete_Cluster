"""Test completo dell'interfaccia neuroni."""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import torch
from src.neurons import LIFNeuron, ALIFNeuron, create_neuron
from src.utils.config_loader import load_config

print("=" * 70)
print("Complete Neuron Interface Integration Test")
print("=" * 70)

# Test 1: Load config and create neurons
print("\n[1] Create neurons from config.yaml:")
config = load_config(root_dir / "configs/config.yaml")

neuron_config = config['neuron']
print(f"✓ Config neuron type: {neuron_config['type']}")

# Create LIF from config
lif = create_neuron(
    n_neurons=256,
    neuron_type=neuron_config['type'],
    config=neuron_config,
)

print(f"✓ LIF created from config")
print(f"  Parameters: tau_mem={neuron_config['lif']['tau_mem']}, "
      f"v_thresh={neuron_config['lif']['v_thresh']}")

# Test 2: Test temporal sequence processing
print("\n[2] Process temporal sequence (like encoder output):")

# Simulate output from PopulationEncoder: (batch, time, 4000)
batch_size = 32
nb_steps = 100
n_input = 4000  # From PopulationEncoder

# Create a sequence of input spikes
spike_input = torch.randint(0, 2, (batch_size, nb_steps, n_input), dtype=torch.float32)
print(f"✓ Input spike sequence: {spike_input.shape}")

# First, collapse to (batch*time, n_input) for processing
spike_input_flat = spike_input.reshape(-1, n_input)

# Initialize neuron
lif_hidden = create_neuron(
    n_neurons=256,
    neuron_type="LIF",
    config=neuron_config,
)

state = lif_hidden.init_state(batch_size, spike_input.device)
hidden_spike_trains = []

# Process each timestep
for t in range(nb_steps):
    # Input for this timestep: (batch, 4000)
    input_t = spike_input[:, t, :]
    
    # Forward through neuron
    spike, state = lif_hidden(input_t, state)
    hidden_spike_trains.append(spike)

hidden_spike_trains = torch.stack(hidden_spike_trains, dim=1)
print(f"✓ Hidden spike train shape: {hidden_spike_trains.shape}")
print(f"  (batch={batch_size}, time={nb_steps}, neurons=256)")
print(f"✓ Hidden spike rate: {hidden_spike_trains.sum() / hidden_spike_trains.numel():.4f}")

# Test 3: Verify state is properly maintained
print("\n[3] State propagation test:")
lif_test = LIFNeuron(n_neurons=100, noise_enabled=False)
state = lif_test.init_state(1, torch.device('cpu'))

# First step
input1 = torch.ones(1, 100) * 0.5
spike1, state1 = lif_test(input1, state)

# Second step with same input, different state
spike2, state2 = lif_test(input1, state1)

# States should be different due to integration
mem_changed = not torch.allclose(state1['mem'], state2['mem'])
print(f"✓ Membrane state propagates: {mem_changed}")
assert mem_changed, "State should change over time!"

# Test 4: Test both neuron types in parallel
print("\n[4] Parallel LIF vs ALIF processing:")
lif_para = LIFNeuron(n_neurons=256, noise_enabled=False)
alif_para = ALIFNeuron(n_neurons=256, noise_enabled=False)

# Use strong input to ensure ALIF generates spikes
strong_input = torch.ones(32, 256) * 0.8

lif_state = lif_para.init_state(32, strong_input.device)
alif_state = alif_para.init_state(32, strong_input.device)

lif_spikes = []
alif_spikes = []

for t in range(50):
    input_t = torch.ones(32, 256) * 0.8 + torch.randn(32, 256) * 0.1
    
    lif_spike, lif_state = lif_para(input_t, lif_state)
    alif_spike, alif_state = alif_para(input_t, alif_state)
    
    lif_spikes.append(lif_spike.sum().item())
    alif_spikes.append(alif_spike.sum().item())

lif_total = sum(lif_spikes)
alif_total = sum(alif_spikes)

print(f"✓ LIF total spikes (50 steps): {int(lif_total)}")
print(f"✓ ALIF total spikes (50 steps): {int(alif_total)}")
print(f"✓ Spike rate LIF: {lif_total / (50*32*256):.4f}")
print(f"✓ Spike rate ALIF: {alif_total / (50*32*256):.4f}")

# Test 5: Verify output is always binary
print("\n[5] Binary output verification:")
lif_test = LIFNeuron(n_neurons=100)
input_test = torch.randn(10, 100)
spike, _ = lif_test(input_test)

unique_vals = torch.unique(spike).numpy()
print(f"✓ Unique values in spike output: {unique_vals}")
assert set(unique_vals).issubset({0.0, 1.0}), "Spike output should be binary!"
print(f"✓ Output is properly binary (0 or 1)")

print("\n" + "=" * 70)
print("✓ ALL INTEGRATION TESTS PASSED!")
print("✓ Neuron interface is fully functional and ready!")
print("=" * 70)

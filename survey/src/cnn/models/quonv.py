"""Quanvolutional layer, ported from `PlanQK/variational-quanvolutional-neural-networks
<https://github.com/PlanQK/variational-quanvolutional-neural-networks>`_ (Mattern et al.,
*Variational Quanvolutional Neural Networks with Enhanced Image Encoding*, 2021).

The original is PennyLane, building one circuit per 2x2 image patch (`Threshold_Encoder` ->
`qml.templates.RandomLayers` -> per-qubit `PauliZ` expectation) and looping over batch/row/col in
plain Python. As with ``cnn/models/qcnn.py``'s port of `takh04/QCNN`, none of that per-sample
looping is load-bearing: for a fixed `calculation_seed` the random-layer ansatz is a *fixed*
sequence of unitaries, so this port builds it once as a dense ``2^4 x 2^4`` real unitary matrix and
applies it to every sliding-window patch of a batch at once as an ordinary matmul.

Ported pieces:

- ``Threshold_Encoder`` (`encoders/threshold_encoder.py`): one qubit per pixel, ``RX(pi)`` if the
  pixel exceeds the threshold else ``RX(0)`` (identity), applied to ``|0>``. Since ``RX`` at those
  two angles only ever maps a basis state to a basis state (never a superposition), the "encoded
  state" for any patch is always exactly one computational basis vector — ported here as directly
  indexing a one-hot vector rather than simulating the rotation.
- ``RandomLayer`` / ``qml.templates.RandomLayers`` (`calculations/random_layer.py`): the
  variational/random ansatz. Its gate *topology* (which wire gets which rotation axis, which pairs
  get CNOT-entangled) is deterministic given ``calculation_seed`` — PennyLane (circa the paper's
  2021 version, pre-v0.29) drives that choice off NumPy's **global legacy RNG**
  (``np.random.seed`` + ``np.random.random``/``np.random.choice``), replicated bit-for-bit in
  ``random_layers_gate_sequence`` below so the same seed produces the same circuit. Only the
  rotation *angles* are the trainable/untrainable ``weights``.
- ``UniformGateMeasurements`` (`measurements/uniform_gate.py`): ``qml.expval(qml.PauliZ(wires=i))``
  on every qubit — one real expectation per qubit, read directly off the batched statevector as
  ``<psi|Z_i|psi>``.
- ``QuonvLayer.convolve`` / ``QNNModel`` (`models/quonv_layer.py`, `models/qnn.py`): the 2x2,
  stride-1, no-padding sliding window, one shared circuit/weights reused at every spatial location
  (weight sharing, like a classical conv kernel), followed by a single flatten + linear head — the
  original never stacks classical conv/pool layers on top.

What's replaced: PennyLane's `default.qubit` per-patch simulation loop is replaced by ``torch.kron``
-built dense real unitaries and a single batched matmul over every patch of the batch at once — the
same restructuring ``qcnn.py`` applies to its own PennyLane source.
"""
import numpy as np
import torch
import torch.nn as nn

N_QUBITS = 4          # one qubit per pixel, Threshold_Encoder on a 2x2 filter
FILTER_SIZE = 2
STRIDE = 1
N_ROTATIONS = 4        # weights.shape[1]: rotation gates placed per RandomLayers layer
N_LAYERS = 1           # weights.shape[0]
RATIO_IMPRIM = 0.4
CALCULATION_SEED = 10   # generate_experiments.py default `calculation_seed`


def random_layers_gate_sequence(seed, n_wires=N_QUBITS, n_layers=N_LAYERS,
                                n_rotations=N_ROTATIONS, ratio_imprim=RATIO_IMPRIM):
    """Replicate `qml.templates.RandomLayers.expand()` (PennyLane <= v0.28, the version the 2021
    paper's code used) bit-for-bit: seeds NumPy's *global* legacy RNG once, then for each layer
    draws gates one at a time until `n_rotations` rotations have been placed, interspersing CNOTs
    on randomly-chosen wire pairs with probability `ratio_imprim` per draw.

    Returns a list of layers; each layer is a list of ops, either
    ``('rot', axis, wire, weight_index)`` with ``axis`` in ``{'RX', 'RY', 'RZ'}``, or
    ``('cnot', control, target)``.
    """
    state = np.random.get_state()   # don't clobber the caller's global numpy RNG state
    try:
        np.random.seed(seed)
        rotations = ['RX', 'RY', 'RZ']
        layers = []
        for _ in range(n_layers):
            ops = []
            i = 0
            while i < n_rotations:
                if np.random.random() > ratio_imprim:
                    axis = np.random.choice(rotations)
                    wire = int(np.random.choice(n_wires, size=1, replace=False)[0])
                    ops.append(('rot', axis, wire, i))
                    i += 1
                elif n_wires > 1:
                    w0, w1 = (int(w) for w in np.random.choice(n_wires, size=2, replace=False))
                    ops.append(('cnot', w0, w1))
            layers.append(ops)
        return layers
    finally:
        np.random.set_state(state)


def _kron_n(mats):
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


def _embed_1q(gate, wire, n_qubits):
    eye = torch.eye(2, dtype=gate.dtype, device=gate.device)
    mats = [eye] * n_qubits
    mats[wire] = gate
    return _kron_n(mats)


def _rx(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    zero = torch.zeros((), dtype=torch.complex128, device=theta.device)
    return torch.stack([
        torch.stack([c.to(torch.complex128), (-1j * s).to(torch.complex128)]),
        torch.stack([(-1j * s).to(torch.complex128), c.to(torch.complex128)]),
    ]) + zero


def _ry(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([
        torch.stack([c, -s]),
        torch.stack([s, c]),
    ]).to(torch.complex128)


def _rz(theta):
    e = torch.exp(-1j * theta.to(torch.complex128) / 2)
    z = torch.zeros((), dtype=torch.complex128, device=theta.device)
    return torch.stack([
        torch.stack([e, z]),
        torch.stack([z, e.conj()]),
    ])


def _cnot(control, target, n_qubits, device):
    dim = 2 ** n_qubits
    eye2 = torch.eye(2, dtype=torch.complex128, device=device)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)
    p0 = torch.zeros(2, 2, dtype=torch.complex128, device=device)
    p0[0, 0] = 1.0
    p1 = torch.zeros(2, 2, dtype=torch.complex128, device=device)
    p1[1, 1] = 1.0

    def embed2(a, wire_a, b, wire_b):
        mats = [eye2] * n_qubits
        mats[wire_a] = a
        mats[wire_b] = b
        return _kron_n(mats)

    return embed2(p0, control, eye2, target) + embed2(p1, control, x, target)


def random_layers_unitary(weights, gate_sequence, n_qubits=N_QUBITS):
    """weights: (n_layers, n_rotations) real tensor -> dense (2^n, 2^n) complex unitary, applying
    `gate_sequence` (from `random_layers_gate_sequence`) in order, each rotation reading its angle
    off `weights[layer, weight_index]`.
    """
    device = weights.device
    dim = 2 ** n_qubits
    U = torch.eye(dim, dtype=torch.complex128, device=device)
    gate_fn = {'RX': _rx, 'RY': _ry, 'RZ': _rz}

    for layer_idx, ops in enumerate(gate_sequence):
        for op in ops:
            if op[0] == 'rot':
                _, axis, wire, widx = op
                gate = gate_fn[axis](weights[layer_idx, widx]).to(torch.complex128)
                U = _embed_1q(gate, wire, n_qubits) @ U
            else:
                _, control, target = op
                U = _cnot(control, target, n_qubits, device) @ U
    return U


def _pauli_z_diag(n_qubits, device):
    """diag[i, b] = <b|Z_i|b> for each computational basis state b, i.e. +1/-1 per qubit."""
    bits = ((torch.arange(2 ** n_qubits, device=device).unsqueeze(1)
             >> torch.arange(n_qubits - 1, -1, -1, device=device).unsqueeze(0)) & 1)
    return 1 - 2 * bits.to(torch.float64)   # (2^n, n_qubits), qubit 0 = most significant


class QuonvLayer(nn.Module):
    """2x2, stride-1 sliding-window quantum convolution: `Threshold_Encoder` state prep -> shared
    `RandomLayers` unitary -> per-qubit `PauliZ` expectation, applied to every patch of the batch
    as one matmul (`QuonvLayer.convolve`/`forward` in the original, minus the per-patch loop).
    """

    def __init__(self, filter_size=FILTER_SIZE, stride=STRIDE, threshold=0.5,
                calculation_seed=CALCULATION_SEED, trainable=True, weights_seed=11111):
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.threshold = threshold
        self.n_qubits = filter_size * filter_size
        self.out_channels = self.n_qubits

        init = np.random.default_rng(weights_seed).uniform(-1, 1, (N_LAYERS, N_ROTATIONS))
        weights = torch.tensor(init, dtype=torch.float64)
        self.trainable = trainable
        if trainable:
            self.weights = nn.Parameter(weights)
        else:
            self.register_buffer('weights', weights)   # frozen at their random initial values

        self.gate_sequence = random_layers_gate_sequence(calculation_seed, n_wires=self.n_qubits)

        # Threshold_Encoder only ever emits a computational basis state (RX(0) or RX(pi) on |0>),
        # so each of the 2^n_qubits thresholded-bit patterns maps to one fixed basis-index.
        n_pixels = self.n_qubits
        patterns = torch.arange(2 ** n_pixels).unsqueeze(1)
        bit_weights = 2 ** torch.arange(n_pixels - 1, -1, -1)
        self.register_buffer('basis_index_of_bits', (patterns // bit_weights.unsqueeze(0)) % 2)
        # basis_index_of_bits[k] is the bit pattern (MSB = pixel/wire 0) for computational index k;
        # inverted below to go from a patch's thresholded bits -> its basis index.
        self.register_buffer('bit_weights', bit_weights)

    def _circuit_unitary(self):
        return random_layers_unitary(self.weights, self.gate_sequence, self.n_qubits)

    def convolve(self, img):
        """img: (B, H, W, 1) in [0, 1] -> patches (B, n_patches_h, n_patches_w, filter_size**2)."""
        patches = img.squeeze(-1).unfold(1, self.filter_size, self.stride) \
                                  .unfold(2, self.filter_size, self.stride)
        b, ph, pw, fh, fw = patches.shape
        return patches.reshape(b, ph, pw, fh * fw)

    def forward(self, img):
        patches = self.convolve(img)                      # (B, ph, pw, n_qubits)
        bits = (patches > self.threshold).to(self.bit_weights.dtype)
        basis_index = (bits * self.bit_weights).sum(-1).long()   # (B, ph, pw)

        U = self._circuit_unitary()                        # (2^n, 2^n)
        z_diag = _pauli_z_diag(self.n_qubits, img.device).to(img.dtype)   # (2^n, n_qubits)

        state = U[:, basis_index.reshape(-1)].T             # (B*ph*pw, 2^n), U @ |basis_index>
        probs = state.abs() ** 2                            # (B*ph*pw, 2^n)
        expvals = probs.to(img.dtype) @ z_diag               # (B*ph*pw, n_qubits)

        b, ph, pw, _ = patches.shape
        return expvals.reshape(b, ph, pw, self.n_qubits)


class QNNModel(nn.Module):
    """`models/qnn.py::QNNModel`: one `QuonvLayer` feeding straight into a single linear head — no
    classical conv/pool/hidden layers in the original.
    """

    def __init__(self, img_size=14, out_features=10, filter_size=FILTER_SIZE, stride=STRIDE,
                calculation_seed=CALCULATION_SEED, trainable=True, weights_seed=11111):
        super().__init__()
        self.qlayer = QuonvLayer(filter_size=filter_size, stride=stride,
                                 calculation_seed=calculation_seed, trainable=trainable,
                                 weights_seed=weights_seed)
        self.flatten = nn.Flatten()
        n_patches = (img_size - filter_size) // stride + 1
        self.linear = nn.Linear(n_patches * n_patches * self.qlayer.out_channels, out_features)

    def forward(self, x):
        """x: (B, 1, H, W) channel-first, matching a torchvision MNIST batch."""
        x = x.permute(0, 2, 3, 1).to(torch.float64)   # -> (B, H, W, 1), matching `predict()`
        x = self.qlayer(x)
        x = self.flatten(x)
        return self.linear(x.to(self.linear.weight.dtype))

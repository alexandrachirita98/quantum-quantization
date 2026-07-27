"""Dressed quantum transfer-learning net, ported from `XanaduAI/quantum-transfer-learning
<https://github.com/XanaduAI/quantum-transfer-learning/blob/master/c2q_transfer_learning_ants_bees.ipynb>`_
(Mari, Bromley, Izaac, Schuld & Killoran, *Transfer learning in hybrid classical-quantum neural
networks*, arXiv:1912.08278, 2019).

The original is PennyLane, calling one 4-qubit `default.qubit` circuit per sample in a Python loop
inside `Quantumnet.forward` (`for elem in q_in: ... torch.cat(...)`) — exactly the per-sample
pattern `cnn/models/qcnn.py` replaces with a batched dense-unitary matmul for `takh04/QCNN`. The
same restructuring applies here, with one twist: unlike QCNN's circuit, this one's *embedding*
layer (`RY(q_in)`) is data-dependent per sample, so only the post-embedding variational unitary is
shared across the batch — the embedding itself is applied as a batched per-wire rotation.

Ported pieces (all from the notebook's cells, verbatim in the docstrings below):

- ``H_layer`` / ``RY_layer`` / ``entangling_layer``: the three circuit building blocks. Notably
  ``entangling_layer`` is a **brick-wall** pattern (``CNOT(0,1), CNOT(2,3)`` then ``CNOT(1,2)`` for
  4 qubits) — *not* a ring (no wraparound ``CNOT(3,0)``), unlike the entanglers elsewhere in this
  codebase.
- ``q_net``: Hadamard-on-every-wire, then ``RY(q_in)`` embedding, then `q_depth` repetitions of
  {entangling_layer; RY(q_weights[k+1])}, then per-wire ``PauliZ`` expectation. Reproduces the
  original's off-by-one: `q_weights` is shaped `(max_layers=15, n_qubits)` but only rows
  `1..q_depth` are ever applied — row 0 is allocated (and consumes from the same `torch.randn`
  draw) but never used, kept here for RNG-consumption fidelity rather than "cleaned up".
- ``Quantumnet`` (named ``Quantumnet`` in the original, not ``DressedQuantumNet``): the "dressed"
  wrapper — ``Linear(512, n_qubits)`` pre-net, ``tanh(.) * pi/2`` squashing into the RY embedding
  angles, the quantum circuit, ``Linear(n_qubits, 2)`` post-net. The pre-net's input width (512)
  matches a frozen ResNet18's global-average-pooled feature size, exactly as the original replaces
  ``resnet18(pretrained=True).fc`` with this module.

What's replaced: PennyLane's `default.qubit` per-sample loop becomes one batched embedding
(per-wire RY applied to every sample's angle at once) followed by one shared dense variational
unitary multiply, mirroring `qcnn.py`'s own departure from its PennyLane source.
"""
import numpy as np
import torch
import torch.nn as nn

N_QUBITS = 4
Q_DEPTH = 6
MAX_LAYERS = 15   # "Keep 15 even if not all are used." -- the original's own comment
Q_DELTA = 0.01


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


def _ry(theta):
    """theta: scalar tensor -> (2, 2) complex RY(theta)."""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([
        torch.stack([c, -s]),
        torch.stack([s, c]),
    ]).to(torch.complex128)


def _cnot(control, target, n_qubits, device):
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


def entangling_unitary(n_qubits=N_QUBITS, device=None, dtype=torch.complex128):
    """`entangling_layer`: CNOTs on even pairs (0,1),(2,3),... then odd pairs (1,2),(3,4),...
    Brick-wall, no wraparound -- for n_qubits=4 this is CNOT(0,1), CNOT(2,3), CNOT(1,2).
    """
    dim = 2 ** n_qubits
    U = torch.eye(dim, dtype=dtype, device=device)
    for i in range(0, n_qubits - 1, 2):
        U = _cnot(i, i + 1, n_qubits, device) @ U
    for i in range(1, n_qubits - 1, 2):
        U = _cnot(i, i + 1, n_qubits, device) @ U
    return U


def _hadamard_state(n_qubits, device, batch=None):
    """H^{otimes n_qubits} |0...0>, i.e. the uniform superposition -- every amplitude 2^{-n/2}."""
    dim = 2 ** n_qubits
    amp = dim ** -0.5
    state = torch.full((dim,), amp, dtype=torch.complex128, device=device)
    return state if batch is None else state.unsqueeze(0).expand(batch, dim).clone()


def ry_embed_batch(state, angles, n_qubits):
    """Apply RY(angles[:, i]) on wire i, for every sample in the batch at once.

    state: (B, 2^n) complex statevectors. angles: (B, n_qubits) real. Since each RY only touches
    one qubit's 2-dim subspace, this is done wire-by-wire as a batched (2,2) block application
    rather than materializing a full (B, 2^n, 2^n) unitary per sample.
    """
    b = state.size(0)
    dim = 2 ** n_qubits
    out = state
    for wire in range(n_qubits):
        theta = angles[:, wire]
        c = torch.cos(theta / 2).to(torch.complex128)
        s = torch.sin(theta / 2).to(torch.complex128)

        view = out.reshape(b, *([2] * n_qubits))
        # indexing wire's axis with an int (not a slice) removes that axis from the result, so
        # a0/a1 have shape (b, *remaining wire dims) -- one fewer dim than `view`.
        idx0 = (slice(None),) * (1 + wire) + (0,) + (slice(None),) * (n_qubits - wire - 1)
        idx1 = (slice(None),) * (1 + wire) + (1,) + (slice(None),) * (n_qubits - wire - 1)
        a0 = view[idx0]
        a1 = view[idx1]

        shape = [b] + [1] * (n_qubits - 1)
        cc = c.reshape(shape)
        ss = s.reshape(shape)
        new_view = view.clone()
        new_view[idx0] = cc * a0 - ss * a1
        new_view[idx1] = ss * a0 + cc * a1
        out = new_view.reshape(b, dim)
    return out


def pauli_z_expvals(state, n_qubits):
    """state: (B, 2^n) complex -> (B, n_qubits) real, <Z_j> for every wire j via marginal probs."""
    b = state.size(0)
    probs = state.abs() ** 2   # (B, 2^n)
    probs = probs.reshape(b, *([2] * n_qubits))
    expvals = []
    for wire in range(n_qubits):
        sum_axes = [1 + q for q in range(n_qubits) if q != wire]
        marginal = probs.sum(dim=sum_axes) if sum_axes else probs   # (B, 2): [P(0), P(1)]
        expvals.append(marginal[:, 0] - marginal[:, 1])
    return torch.stack(expvals, dim=1)


def q_net(q_in, q_weights_flat, n_qubits=N_QUBITS, q_depth=Q_DEPTH, max_layers=MAX_LAYERS):
    """Batched port of the original's per-sample `q_net` qnode.

    q_in: (B, n_qubits) real, already tanh*(pi/2)-squashed embedding angles.
    q_weights_flat: (max_layers * n_qubits,) real, the flat trainable `q_params`.
    Returns (B, n_qubits) real PauliZ expectations, matching
    `[qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]`.
    """
    device = q_in.device
    b = q_in.size(0)
    q_weights = q_weights_flat.reshape(max_layers, n_qubits)

    state = _hadamard_state(n_qubits, device, batch=b)
    state = ry_embed_batch(state, q_in.to(torch.complex128).real, n_qubits)

    U_ent = entangling_unitary(n_qubits, device)
    for k in range(q_depth):
        state = (U_ent.unsqueeze(0) @ state.unsqueeze(-1)).squeeze(-1)
        state = ry_embed_batch(state, q_weights[k + 1].unsqueeze(0).expand(b, -1), n_qubits)

    return pauli_z_expvals(state, n_qubits)


class Quantumnet(nn.Module):
    """The "dressed quantum circuit" replacing `resnet18(pretrained=True).fc`.

    `pre_net`: Linear(512, n_qubits) mapping a frozen ResNet18's pooled features down to the
    embedding width. `q_params`: the flat variational weight vector, `q_delta`-scaled random init
    (matches `q_delta * torch.randn(max_layers * n_qubits)` exactly, including the never-used row 0
    of the reshaped `(max_layers, n_qubits)` view). `post_net`: Linear(n_qubits, 2) producing the
    2-class (ants vs. bees) logits fed to `nn.CrossEntropyLoss`.
    """

    def __init__(self, n_qubits=N_QUBITS, q_depth=Q_DEPTH, max_layers=MAX_LAYERS,
                q_delta=Q_DELTA, in_features=512, out_features=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.max_layers = max_layers

        self.pre_net = nn.Linear(in_features, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(max_layers * n_qubits))
        self.post_net = nn.Linear(n_qubits, out_features)

    def forward(self, input_features):
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        q_out = q_net(q_in, self.q_params, self.n_qubits, self.q_depth, self.max_layers)
        return self.post_net(q_out.to(self.post_net.weight.dtype))

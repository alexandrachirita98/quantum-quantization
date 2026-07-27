"""QCNN, ported from `takh04/QCNN <https://github.com/takh04/QCNN>`_ (Hur, Kim & Park,
*Quantum convolutional neural network for classical data classification*, 2022).

The original repo is PennyLane, simulating one 8-qubit ``default.qubit`` state per sample in a
Python loop and training with ``NesterovMomentumOptimizer`` via PennyLane's own autograd. Neither
choice is load-bearing: the circuit itself is a fixed sequence of unitaries, so this port builds
each gate as a dense ``2^8 x 2^8`` complex matrix, applies it to a *batch* of statevectors as an
ordinary matrix multiply, and trains with plain ``torch.autograd`` + Adam — exactly the departure
``vae/models/qvae.py`` takes from its own Qiskit source.

Ported pieces (``unitary.py`` / ``QCNN_circuit.py`` in the original):

- ``U_SU4`` (15 params): the general two-qubit convolution ansatz the repo's benchmarking run
  uses as its primary unitary. ``U3(theta, phi, lambda)`` is expanded as ``RZ(phi) RY(theta)
  RZ(lambda)``, matching the standard up-to-global-phase identity.
- ``Pooling_ansatz1`` (2 params): ``CRZ(p0, [w0,w1]); PauliX(w0); CRX(p1, [w0,w1])``, the only
  pooling ansatz the original ever actually calls. Wires are ``[discarded, kept]`` — pooling does
  *not* trace anything out; the discarded wire is just never read from or acted on again, so a
  literal port never needs a partial trace.
- The standard ``QCNN_structure`` topology: 3 conv+pool stages reducing 8 qubits to 4, to 2, to 1,
  each conv layer *reusing one shared parameter vector* across every wire pair it touches (the
  circuit's analogue of a classical conv kernel), ending on wire 4 (not wire 0).

Readout is ``qml.probs(wires=4)`` in the original's cross-entropy mode — the marginal
[P(|0>), P(|1>)] of the final surviving qubit, obtained here from ``|amplitude|^2`` summed over
the other 7 qubits.
"""
import torch
import torch.nn as nn

N_QUBITS = 8
U_SU4_PARAMS = 15
POOL_PARAMS = 2


def _kron_n(mats):
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


def _embed_1q(gate, wire, n_qubits):
    """gate: (2, 2) complex tensor -> its (2^n, 2^n) embedding acting on `wire`."""
    eye = torch.eye(2, dtype=gate.dtype, device=gate.device)
    mats = [eye] * n_qubits
    mats[wire] = gate
    return _kron_n(mats)


def _rx(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    zero = torch.zeros_like(c)
    return torch.stack([
        torch.stack([c.to(torch.complex128), (-1j * s).to(torch.complex128)]),
        torch.stack([(-1j * s).to(torch.complex128), c.to(torch.complex128)]),
    ])


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


def _u3(theta, phi, lam):
    """U3(theta, phi, lambda) = RZ(phi) . RY(theta) . RZ(lambda), up to global phase."""
    return _rz(phi) @ _ry(theta).to(torch.complex128) @ _rz(lam)


def _hadamard(device):
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128, device=device) / (2 ** 0.5)
    return h


def _two_qubit_embed(mat2x2_kron_target, control, target, n_qubits, control_proj, target_gate):
    """Build a controlled-U: |0><0|_c (x) I_t + |1><1|_c (x) U_t, embedded in n_qubits."""
    dim = 2 ** n_qubits
    eye_full = torch.eye(dim, dtype=torch.complex128, device=target_gate.device)
    p0 = torch.zeros(2, 2, dtype=torch.complex128, device=target_gate.device)
    p0[0, 0] = 1.0
    p1 = torch.zeros(2, 2, dtype=torch.complex128, device=target_gate.device)
    p1[1, 1] = 1.0
    eye2 = torch.eye(2, dtype=torch.complex128, device=target_gate.device)

    def embed2(a, wire_a, b, wire_b):
        mats = [eye2] * n_qubits
        mats[wire_a] = a
        mats[wire_b] = b
        return _kron_n(mats)

    term0 = embed2(p0, control, eye2, target)
    term1 = embed2(p1, control, target_gate, target)
    return term0 + term1


def _cnot(control, target, n_qubits, device):
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)
    return _two_qubit_embed(None, control, target, n_qubits, None, x)


def _crz(theta, control, target, n_qubits):
    return _two_qubit_embed(None, control, target, n_qubits, None, _rz(theta))


def _crx(theta, control, target, n_qubits):
    return _two_qubit_embed(None, control, target, n_qubits, None, _rx(theta))


def _cz(control, target, n_qubits, device):
    z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)
    return _two_qubit_embed(None, control, target, n_qubits, None, z)


def u_su4(params, wires, n_qubits):
    """15-parameter general two-qubit unitary, `unitary.py::U_SU4`.

    U3(p0,p1,p2,w0); U3(p3,p4,p5,w1); CNOT(w0,w1); RY(p6,w0); RZ(p7,w1); CNOT(w1,w0);
    RY(p8,w0); CNOT(w0,w1); U3(p9,p10,p11,w0); U3(p12,p13,p14,w1)
    """
    w0, w1 = wires
    device = params.device
    dim = 2 ** n_qubits
    U = torch.eye(dim, dtype=torch.complex128, device=device)

    U = _embed_1q(_u3(params[0], params[1], params[2]), w0, n_qubits) @ U
    U = _embed_1q(_u3(params[3], params[4], params[5]), w1, n_qubits) @ U
    U = _cnot(w0, w1, n_qubits, device) @ U
    U = _embed_1q(_ry(params[6]).to(torch.complex128), w0, n_qubits) @ U
    U = _embed_1q(_rz(params[7]), w1, n_qubits) @ U
    U = _cnot(w1, w0, n_qubits, device) @ U
    U = _embed_1q(_ry(params[8]).to(torch.complex128), w0, n_qubits) @ U
    U = _cnot(w0, w1, n_qubits, device) @ U
    U = _embed_1q(_u3(params[9], params[10], params[11]), w0, n_qubits) @ U
    U = _embed_1q(_u3(params[12], params[13], params[14]), w1, n_qubits) @ U
    return U


def pooling_ansatz1(params, wires, n_qubits):
    """2-parameter pooling unitary, `unitary.py::Pooling_ansatz1`.

    CRZ(p0, [w0,w1]); PauliX(w0); CRX(p1, [w0,w1]). `w0` is the wire being discarded (control),
    `w1` the wire being kept (target) — the original never traces `w0` out, it simply stops
    addressing it, so this port carries the full statevector throughout.
    """
    w0, w1 = wires
    device = params.device
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)

    U = _crz(params[0], w0, w1, n_qubits)
    U = _embed_1q(x, w0, n_qubits) @ U
    U = _crx(params[1], w0, w1, n_qubits) @ U
    return U


class QCNN(nn.Module):
    """8-qubit amplitude-encoded QCNN: 3 conv+pool stages, 8 -> 4 -> 2 -> 1 qubits.

    Matches `QCNN_circuit.py::QCNN_structure`: each conv layer reuses one shared 15-parameter
    ``U_SU4`` across every wire pair it touches (weight sharing, like a classical conv kernel);
    each pooling layer reuses one shared 2-parameter ``Pooling_ansatz1``. Readout is the marginal
    probability of the final surviving qubit (wire 4) being |0> or |1>, matching
    ``qml.probs(wires=4)`` under the original's ``cost_fn='cross_entropy'`` mode.
    """

    def __init__(self, n_qubits=N_QUBITS):
        super().__init__()
        assert n_qubits == 8, "the wire layout below is hardcoded for 8 qubits, as in the original"
        self.n_qubits = n_qubits
        self.readout_wire = 4

        self.conv1 = nn.Parameter(0.1 * torch.randn(U_SU4_PARAMS, dtype=torch.float64))
        self.conv2 = nn.Parameter(0.1 * torch.randn(U_SU4_PARAMS, dtype=torch.float64))
        self.conv3 = nn.Parameter(0.1 * torch.randn(U_SU4_PARAMS, dtype=torch.float64))
        self.pool1 = nn.Parameter(0.1 * torch.randn(POOL_PARAMS, dtype=torch.float64))
        self.pool2 = nn.Parameter(0.1 * torch.randn(POOL_PARAMS, dtype=torch.float64))
        self.pool3 = nn.Parameter(0.1 * torch.randn(POOL_PARAMS, dtype=torch.float64))

        # conv_layer1: same params applied on wire pairs (0,7),(0,1),(2,3),(4,5),(6,7),(1,2),(3,4),(5,6)
        self.conv1_pairs = [(0, 7), (0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6)]
        # pooling_layer1: [discarded, kept] = (1,0),(3,2),(5,4),(7,6) -> survivors {0,2,4,6}
        self.pool1_pairs = [(1, 0), (3, 2), (5, 4), (7, 6)]
        # conv_layer2 on survivors {0,2,4,6}
        self.conv2_pairs = [(0, 6), (0, 2), (4, 6), (2, 4)]
        # pooling_layer2: (2,0),(6,4) -> survivors {0,4}
        self.pool2_pairs = [(2, 0), (6, 4)]
        # conv_layer3 on survivors {0,4}
        self.conv3_pairs = [(0, 4)]
        # pooling_layer3: (0,4) -> survivor {4}
        self.pool3_pairs = [(0, 4)]

    def _full_unitary(self):
        n = self.n_qubits
        dim = 2 ** n
        U = torch.eye(dim, dtype=torch.complex128, device=self.conv1.device)

        for pair in self.conv1_pairs:
            U = u_su4(self.conv1, pair, n) @ U
        for pair in self.pool1_pairs:
            U = pooling_ansatz1(self.pool1, pair, n) @ U

        for pair in self.conv2_pairs:
            U = u_su4(self.conv2, pair, n) @ U
        for pair in self.pool2_pairs:
            U = pooling_ansatz1(self.pool2, pair, n) @ U

        for pair in self.conv3_pairs:
            U = u_su4(self.conv3, pair, n) @ U
        for pair in self.pool3_pairs:
            U = pooling_ansatz1(self.pool3, pair, n) @ U

        return U

    def forward(self, x):
        """x: (B, 2^n_qubits) real or complex amplitude-encoded, L2-normalized states.

        Returns (B, 2) marginal probabilities [P(wire4=0), P(wire4=1)], matching
        `qml.probs(wires=4)`.
        """
        x = x.to(torch.complex128)
        U = self._full_unitary()
        out = x @ U.T  # (B, dim), since U acts as `U @ x` per-sample: x_row @ U.T

        n = self.n_qubits
        probs = out.abs() ** 2  # (B, 2^n)
        probs = probs.reshape(-1, *([2] * n))
        # sum out every wire except the readout wire, matching qml.probs(wires=[readout_wire])
        keep_axis = 1 + self.readout_wire  # axis 0 is batch
        sum_axes = [1 + q for q in range(n) if q != self.readout_wire]
        marginal = probs.sum(dim=sum_axes)  # (B, 2), still ordered [batch, wire]
        return marginal

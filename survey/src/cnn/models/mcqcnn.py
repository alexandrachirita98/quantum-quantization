"""Multi-Channel QCNN, ported from `anthonysmaldone/QCNN-Multi-Channel-Supervised-Learning
<https://github.com/anthonysmaldone/QCNN-Multi-Channel-Supervised-Learning>`_ (Smaldone & Batista,
*Hardware-adaptable quantum circuits for convolutional layers with multi-channel data*, Quantum
Machine Intelligence, 2023, https://doi.org/10.1007/s42484-023-00130-3).

The original is TensorFlow + TensorFlow Quantum + Cirq: a sliding 2x2 quantum-convolution kernel
(à la a classical `Conv2D`) simulated via `tfq.layers.Expectation()`, which is TF-autodiff
differentiable end to end — the same "backprop through the simulator, no manual parameter-shift"
situation `cnn/models/qcnn.py` and `cnn/models/qtransfer.py` exploit for their own PennyLane
sources. This port replaces `tfq.layers.Expectation()` with dense complex-matrix construction and a
batched matmul, the same restructuring applied throughout this codebase.

**This is architecturally different from every other quantum notebook in this folder.** Where
`qcnn.py` is one fixed 8-qubit conv+pool pyramid and `quonv.py` is a single-filter sliding window,
this repo defines **four separate circuit families**, each a genuine sliding 2x2 kernel (like a
classical `Conv2D`, stride 1, no padding, `n_kernels` independently-learned parameter sets = output
feature maps), differing in how multi-channel data is combined:

- ``U1Circuit`` / ``U2Circuit`` (`circuits.py::U1_circuit` / `U2_circuit`) — the paper's proposed
  "multi-channel-aware" ansatzes. Every channel gets its own 4-qubit *data register*, angle-encoded
  in parallel (or in sequential waves, if there are more channels than registers), then **entangled
  with each other in-circuit** (``inter_U``) and made to deposit phase onto one or more shared
  *ancilla* qubit(s) before a single ``PauliX`` readout on ancilla 0. Channel combination happens
  *inside* the quantum circuit. ``U1`` and ``U2`` differ only in their intra-/inter-channel
  entangler gate sequences (``U2`` adds extra ``CZPowGate`` terms alongside every ``CXPowGate``).
- ``QU1Control`` / ``QU2Control`` (`circuits.py::Q_U1_control` / `Q_U2_control`) — the paper's
  non-multi-channel-aware *baselines*: a single fixed 4-qubit circuit (no ancilla) run
  independently once per channel with its own private parameter set, ``PauliZ``-read on qubit 0,
  and combined *classically* afterward via a plain sum (or, with ``classical_weights=True``, a
  learned per-channel affine reweight then sum — the paper's "WEV" variant).

Every entangling gate is a **fractional power of X or Z** (Cirq's ``CXPowGate``/``CZPowGate``,
continuously interpolating identity <-> CNOT/CZ via a trainable exponent), not a fixed-angle
rotation like the ``RY``/``RZ``/``U3`` gates elsewhere in this codebase. Reproduced here as dense
matrices matching Cirq's exact convention (verified against `cirq.unitary(...)`):
``ZPowGate(t) = diag(1, e^{i pi t})``, ``XPowGate(t) = e^{i pi t/2} [[cos(pi t/2), -i sin(pi t/2)],
[-i sin(pi t/2), cos(pi t/2)]]``, and the controlled versions act as identity on ``|0><0|`` of the
control and as the 1-qubit gate on ``|1><1|`` of the control -- ``GATE(exponent=t)(q_control,
q_target)`` in the original's calling convention.

Every circuit also applies the same post-measurement "un-embedding" nonlinearity before its
classical head: ``arccos(clip(expectation, -1+1e-5, 1-1e-5)) / pi``, mapping the ``[-1, 1]``
expectation back to a ``[0, 1]`` angle -- this is not optional, it is part of every `call()` in the
original and is reproduced here in ``un_embed``.
"""
import numpy as np
import torch
import torch.nn as nn

FILTER_SIZE = 2
N_FILTER_QUBITS = FILTER_SIZE * FILTER_SIZE   # one qubit per pixel of the 2x2 patch


# --------------------------------------------------------------------------- #
# Dense gate primitives, matching Cirq's exact conventions
# --------------------------------------------------------------------------- #
def x_pow(theta):
    """cirq.XPowGate(exponent=t), t = theta/pi -- theta is passed in radians (t = theta / pi)."""
    t = theta
    c = torch.cos(np.pi * t / 2)
    s = torch.sin(np.pi * t / 2)
    g = torch.exp(1j * np.pi * t / 2).to(torch.complex128)
    c, s = c.to(torch.complex128), s.to(torch.complex128)
    return torch.stack([
        torch.stack([g * c, -1j * g * s]),
        torch.stack([-1j * g * s, g * c]),
    ])


def z_pow(t):
    """cirq.ZPowGate(exponent=t) = diag(1, e^{i pi t})."""
    one = torch.ones((), dtype=torch.complex128, device=t.device)
    e = torch.exp(1j * np.pi * t.to(torch.complex128))
    zero = torch.zeros((), dtype=torch.complex128, device=t.device)
    return torch.stack([
        torch.stack([one, zero]),
        torch.stack([zero, e]),
    ])


def rx_embed(theta):
    """cirq.rx(theta) = XPowGate(exponent=theta/pi, global_shift=-0.5) -- Cirq's `rx` convention
    differs from `XPowGate` by a global phase (`global_shift=-0.5`), matching `cirq.rx`'s
    definition as exp(-i*theta*X/2) exactly (no extra phase), used for the data-embedding rotation.
    """
    c = torch.cos(theta / 2).to(torch.complex128)
    s = torch.sin(theta / 2).to(torch.complex128)
    return torch.stack([
        torch.stack([c, -1j * s]),
        torch.stack([-1j * s, c]),
    ])


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


def _embed_controlled_1q(gate1q, control, target, n_qubits):
    """GATE(exponent=t)(q_control, q_target): identity on control=|0>, `gate1q` on target when
    control=|1>. Matches Cirq's `CXPowGate`/`CZPowGate` positional convention -- first qubit
    argument is the control.
    """
    eye2 = torch.eye(2, dtype=torch.complex128, device=gate1q.device)
    p0 = torch.zeros(2, 2, dtype=torch.complex128, device=gate1q.device)
    p0[0, 0] = 1.0
    p1 = torch.zeros(2, 2, dtype=torch.complex128, device=gate1q.device)
    p1[1, 1] = 1.0

    def embed2(a, wire_a, b, wire_b):
        mats = [eye2] * n_qubits
        mats[wire_a] = a
        mats[wire_b] = b
        return _kron_n(mats)

    return embed2(p0, control, eye2, target) + embed2(p1, control, gate1q, target)


def cx_pow(theta, control, target, n_qubits):
    return _embed_controlled_1q(x_pow(theta), control, target, n_qubits)


def cz_pow(theta, control, target, n_qubits):
    return _embed_controlled_1q(z_pow(theta), control, target, n_qubits)


def _apply_controlled_1q_batch(state, gate1q, control, target, n_qubits):
    """Apply `_embed_controlled_1q(gate1q, control, target, n_qubits)` to a batch of statevectors
    *without materializing the (2^n, 2^n) matrix* -- same batched-per-wire trick as
    ``_rx_embed_batch``/``_pauli_op_expval`` above, and mathematically identical to
    ``_embed_controlled_1q(...) @ state``.

    This matters for the ``registers=3`` (PCO / PCO-T) circuits: those run on 13 and 15 qubits, so
    one dense gate would be 8192^2 (1 GB) and 32768^2 (17 GB) complex128 respectively -- per gate,
    per kernel, all retained by autograd for the backward pass. The control baselines stay on the
    dense path since they are only 4 qubits (16x16).

    The gate acts as identity on the ``control=|0>`` half of the state, and as the 1-qubit
    ``gate1q`` on the ``target`` axis within the ``control=|1>`` half -- Cirq's positional
    convention, where the first qubit argument is the control.
    """
    b = state.size(0)
    dim = 2 ** n_qubits

    def sel(c_val, t_val):
        idx = [slice(None)] * (1 + n_qubits)
        idx[1 + control] = c_val
        idx[1 + target] = t_val
        return tuple(idx)

    view = state.reshape(b, *([2] * n_qubits))
    a0 = view[sel(1, 0)]
    a1 = view[sel(1, 1)]

    new_view = view.clone()
    new_view[sel(1, 0)] = gate1q[0, 0] * a0 + gate1q[0, 1] * a1
    new_view[sel(1, 1)] = gate1q[1, 0] * a0 + gate1q[1, 1] * a1
    return new_view.reshape(b, dim)


def un_embed(expectation):
    """`arccos(clip(e, -1+1e-5, 1-1e-5)) / pi`, applied before the classical head in every
    circuit's `call()` in the original.
    """
    return torch.arccos(expectation.clamp(-1 + 1e-5, 1 - 1e-5)) / np.pi


def extract_patches(images, filter_size=FILTER_SIZE, stride=1):
    """images: (B, H, W, C) -> (B, num_x, num_y, C, filter_size, filter_size), matching the
    original's `call()` slide-and-stack loop (stride 1, no padding).
    """
    b, h, w, c = images.shape
    patches = images.permute(0, 3, 1, 2).unfold(2, filter_size, stride).unfold(3, filter_size, stride)
    # patches: (B, C, num_x, num_y, filter_size, filter_size)
    return patches.permute(0, 2, 3, 1, 4, 5)


# --------------------------------------------------------------------------- #
# U1 / U2: multi-channel-aware circuits (channel combination happens in-circuit)
# --------------------------------------------------------------------------- #
class _MultiChannelCircuit(nn.Module):
    """Shared machinery for `U1Circuit`/`U2Circuit`: register/ancilla bookkeeping, the batched
    statevector simulation, and the `PauliX`-on-ancilla-0 readout. Subclasses only need to supply
    the gate-placement schedule via `_build_gate_schedule`.
    """

    def __init__(self, n_kernels, n_input_channels, registers=1, rdpa=1, inter_U=False):
        super().__init__()
        self.n_kernels = n_kernels
        self.n_input_channels = n_input_channels
        self.registers = registers
        self.rdpa = rdpa
        self.ancilla = registers // rdpa
        self.inter_U = inter_U
        self.circuit_layers = -(-n_input_channels // registers)   # ceil division

        self.n_qubits = self.ancilla + registers * N_FILTER_QUBITS
        # wire layout: [0 .. ancilla-1] ancilla, then register r occupies
        # [ancilla + r*4 .. ancilla + r*4 + 3]
        self.gate_schedule, self.n_params = self._build_gate_schedule()

        self.kernel = nn.Parameter(_glorot_normal((n_kernels, 1, self.n_params)))

    def _register_wires(self, r):
        base = self.ancilla + r * N_FILTER_QUBITS
        return list(range(base, base + N_FILTER_QUBITS))

    def _build_gate_schedule(self):
        raise NotImplementedError

    def forward(self, images):
        """images: (B, H, W, n_input_channels) in [0, 1]. Returns (B, num_x, num_y, n_kernels)
        after the `arccos/pi` un-embedding (activation is applied by the caller, matching the
        original's `self.activation(output_tensor)` being outside the un-embed step... actually
        matches the original which applies activation AFTER un-embed, see `MCQCNNModel`).
        """
        b, h, w, c = images.shape
        patches = extract_patches(images)   # (B, num_x, num_y, C, 2, 2)
        num_x, num_y = patches.shape[1], patches.shape[2]
        pixels = patches.reshape(b, num_x, num_y, c, N_FILTER_QUBITS)   # last dim = 4 pixels/channel

        device = images.device
        outputs = []
        for k in range(self.n_kernels):
            state = self._run_circuit(pixels, self.kernel[k, 0], device)   # (B, num_x, num_y, 2^n)
            expval = _pauli_op_expval(state, 0, pauli='X', n_qubits=self.n_qubits)
            outputs.append(expval)
        stacked = torch.stack(outputs, dim=-1)   # (B, num_x, num_y, n_kernels)
        return un_embed(stacked)

    def _run_circuit(self, pixels, kernel_params, device):
        b, num_x, num_y = pixels.shape[0], pixels.shape[1], pixels.shape[2]
        n = self.n_qubits
        dim = 2 ** n
        flat_b = b * num_x * num_y

        state = _uniform_ancilla_state(self.ancilla, self.registers, N_FILTER_QUBITS, device)
        state = state.unsqueeze(0).expand(flat_b, dim).clone()

        pixels_flat = pixels.reshape(flat_b, self.n_input_channels, N_FILTER_QUBITS)

        param_iter = iter(kernel_params)
        for layer in range(self.circuit_layers):
            # embed
            for r in range(self.registers):
                ch = r + layer * self.registers
                if ch >= self.n_input_channels:
                    continue
                wires = self._register_wires(r)
                angles = np.pi * pixels_flat[:, ch, :]   # (flat_b, 4)
                state = _rx_embed_batch(state, angles, wires, n)

            # gate schedule for this layer: list of ('rx'-independent) ops already resolved to wires
            state = self._apply_schedule_ops(state, kernel_params, n)

        return state.reshape(b, num_x, num_y, dim)

    def _apply_schedule_ops(self, state, kernel_params, n):
        for op in self.gate_schedule:
            kind, wire_a, wire_b, pidx = op
            theta = kernel_params[pidx]
            gate1q = x_pow(theta) if kind == 'cx' else z_pow(theta)
            state = _apply_controlled_1q_batch(state, gate1q, wire_a, wire_b, n)
        return state


def _glorot_normal(shape):
    fan_in, fan_out = shape[-2] if len(shape) > 1 else shape[-1], shape[-1]
    std = (2.0 / (fan_in + fan_out)) ** 0.5
    return torch.randn(shape, dtype=torch.float64) * std


def _uniform_ancilla_state(ancilla, registers, per_register, device):
    """H on every ancilla wire, |0> elsewhere -- `cirq.H` applied before any embedding."""
    n = ancilla + registers * per_register
    dim = 2 ** n
    state = torch.zeros(dim, dtype=torch.complex128, device=device)
    state[0] = 1.0
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128, device=device) / 2 ** 0.5
    for w in range(ancilla):
        state = _embed_1q(h, w, n) @ state
    return state


def _rx_embed_batch(state, angles, wires, n_qubits):
    """Apply `rx(angles[:, i])` on `wires[i]` for every sample in the batch, in place-ish (no full
    per-sample unitary materialized -- same batched-per-wire trick as `qtransfer.py::ry_embed_batch`).
    """
    b = state.size(0)
    dim = 2 ** n_qubits
    out = state
    for i, wire in enumerate(wires):
        theta = angles[:, i]
        c = torch.cos(theta / 2).to(torch.complex128)
        s = torch.sin(theta / 2).to(torch.complex128)

        view = out.reshape(b, *([2] * n_qubits))
        idx0 = (slice(None),) * (1 + wire) + (0,) + (slice(None),) * (n_qubits - wire - 1)
        idx1 = (slice(None),) * (1 + wire) + (1,) + (slice(None),) * (n_qubits - wire - 1)
        a0 = view[idx0]
        a1 = view[idx1]

        shape = [b] + [1] * (n_qubits - 1)
        cc = c.reshape(shape)
        ss = s.reshape(shape)
        new_view = view.clone()
        new_view[idx0] = cc * a0 - 1j * ss * a1
        new_view[idx1] = -1j * ss * a0 + cc * a1
        out = new_view.reshape(b, dim)
    return out


def _pauli_op_expval(state, wire, pauli, n_qubits):
    """<psi|P_wire|psi> for P in {X, Z}, computed from the (flattened batch, 2^n) statevector
    without materializing the full (2^n, 2^n) Pauli matrix.
    """
    flat_b = state.shape[:-1]
    dim = state.shape[-1]
    view = state.reshape(*flat_b, *([2] * n_qubits))
    idx0 = (Ellipsis, 0) + (slice(None),) * (n_qubits - wire - 1)
    idx1 = (Ellipsis, 1) + (slice(None),) * (n_qubits - wire - 1)
    idx0 = (slice(None),) * wire + idx0
    idx1 = (slice(None),) * wire + idx1
    a0 = view[idx0].reshape(*flat_b, -1)
    a1 = view[idx1].reshape(*flat_b, -1)
    if pauli == 'Z':
        return (a0.abs() ** 2).sum(-1) - (a1.abs() ** 2).sum(-1)
    # X: <psi|X|psi> = 2*Re(conj(a0)*a1) summed over the rest of the state
    return 2 * (a0.conj() * a1).sum(-1).real


def _entangle_pair(schedule, params_counter, source, target, wires_src, wires_tgt, kind='cx'):
    idx = next(params_counter)
    schedule.append((kind, wires_tgt[target], wires_src[source], idx))


class U1Circuit(_MultiChannelCircuit):
    """`circuits.py::U1_circuit`. Intra-channel: a 4-cycle of `CXPowGate` (1->0, 2->1, 3->2, 0->3).
    Inter-channel (if `inter_U`): a ring of `CXPowGate` between register-0 wires. Deposit: one
    `CZPowGate` from wire 0 of each data register onto its assigned ancilla. Ancilla-entangle (if
    `ancilla>1`): a ring of `CXPowGate` between ancilla wires.
    """

    def _build_gate_schedule(self):
        schedule = []
        counter = iter(range(10 ** 9))

        def register_wires(r):
            base = self.ancilla + r * N_FILTER_QUBITS
            return list(range(base, base + N_FILTER_QUBITS))

        for layer in range(self.circuit_layers):
            # intra-channel entangle, every register
            for r in range(self.registers):
                w = register_wires(r)
                _entangle_pair(schedule, counter, 1, 0, w, w)
                _entangle_pair(schedule, counter, 2, 1, w, w)
                _entangle_pair(schedule, counter, 3, 2, w, w)
                _entangle_pair(schedule, counter, 0, 3, w, w)

            # inter-channel entangle (registers > 1 and inter_U): a ring, control=reg[i+1][0]
            # (wrapping the last back to reg[0]), rotated=reg[i][0] -- matches the original's
            # `for i in range(1, len(qubits_all)): Q_new_entangle(0, 0, qubits_all[i+1 or wrap],
            # qubits_all[i])` where `Q_new_entangle(source, target, qubits_tar, qubits_src)` gates
            # `(qubits_tar[target], qubits_src[source])` = `(control, rotated)`; `_entangle_pair`'s
            # `wires_tgt`/`wires_src` params play the `qubits_tar`/`qubits_src` roles respectively
            # (qubits_all is 1-indexed with ancilla at 0; reg_wires here is 0-indexed data-only).
            if self.registers > 1 and self.inter_U:
                reg_wires = [register_wires(r) for r in range(self.registers)]
                if self.registers > 2:
                    for i in range(self.registers):
                        j = i + 1 if i != self.registers - 1 else 0
                        _entangle_pair(schedule, counter, 0, 0, reg_wires[i], reg_wires[j])
                else:
                    # original: `Q_new_entangle(0, 0, qubits_all[1], qubits_all[2])` -- only ever
                    # reached when `registers == 2` in the original (registers<=1 skips this whole
                    # block via the `self.registers > 1` guard above), so `qubits_all[2]` is valid
                    # (it is 1-indexed data register 2 of 2). qubits_tar=qubits_all[1]=reg_wires[0],
                    # qubits_src=qubits_all[2]=reg_wires[1].
                    _entangle_pair(schedule, counter, 0, 0, reg_wires[1], reg_wires[0])

            # deposit onto ancilla, round-robin: original `CZPowGate(...)(qubits[0], ancilla)`,
            # i.e. control=register wire 0, rotated=ancilla -- but CZPowGate's matrix is symmetric
            # in its two qubits (diag(1,1,1,e^{i pi t})), so which one is labeled "control" here
            # does not change the resulting unitary.
            ancilla_count = 1
            for r in range(self.registers):
                w = register_wires(r)
                anc = ancilla_count - 1
                idx = next(counter)
                schedule.append(('cz', anc, w[0], idx))
                if ancilla_count < self.ancilla:
                    ancilla_count += 1

            # ancilla-entangle: original `Q_ancilla_entangle` -- for i in range(ancilla): if
            # i != ancilla-1: Q_new_entangle(source=i, target=i+1) (control=qubits[i+1],
            # rotated=qubits[i]); else: Q_new_entangle(source=0, target=i) (control=qubits[i],
            # rotated=qubits[0]) -- the last iteration swaps source/target, not just the wire index.
            if self.registers > 1 and self.ancilla > 1:
                anc_wires = list(range(self.ancilla))
                if self.ancilla > 2:
                    for i in range(self.ancilla):
                        if i != self.ancilla - 1:
                            _entangle_pair(schedule, counter, i, i + 1, anc_wires, anc_wires)
                        else:
                            _entangle_pair(schedule, counter, 0, i, anc_wires, anc_wires)
                else:
                    _entangle_pair(schedule, counter, 0, 1, anc_wires, anc_wires)

        n_params = max((op[3] for op in schedule), default=-1) + 1
        return schedule, n_params


class U2Circuit(_MultiChannelCircuit):
    """`circuits.py::U2_circuit`. Intra-channel: 3 pairs (1->0, 3->2, 2->0), each with **both** a
    `CZPowGate` and a `CXPowGate` (2 params/edge instead of U1's 1). Inter-channel/ancilla-entangle:
    same topology as U1 but `CXPowGate` only (no accompanying `CZPowGate`).
    """

    def _build_gate_schedule(self):
        schedule = []
        counter = iter(range(10 ** 9))

        def register_wires(r):
            base = self.ancilla + r * N_FILTER_QUBITS
            return list(range(base, base + N_FILTER_QUBITS))

        def entangle_cz_then_cx(source, target, wires_tgt, wires_src, cz=True):
            if cz:
                _entangle_pair(schedule, counter, source, target, wires_src, wires_tgt, kind='cz')
            _entangle_pair(schedule, counter, source, target, wires_src, wires_tgt, kind='cx')

        for layer in range(self.circuit_layers):
            for r in range(self.registers):
                w = register_wires(r)
                entangle_cz_then_cx(1, 0, w, w)
                entangle_cz_then_cx(3, 2, w, w)
                entangle_cz_then_cx(2, 0, w, w)

            # same ring topology (and the same off-by-one-prone indexing) as `U1Circuit`'s
            # inter-channel entangle -- see the detailed comment there. `entangle_cz_then_cx`'s
            # (source, target, wires_tgt, wires_src) mirrors `Q_new_entangle`'s own argument order.
            if self.registers > 1 and self.inter_U:
                reg_wires = [register_wires(r) for r in range(self.registers)]
                if self.registers > 2:
                    for i in range(self.registers):
                        j = i + 1 if i != self.registers - 1 else 0
                        entangle_cz_then_cx(0, 0, reg_wires[j], reg_wires[i], cz=False)
                else:
                    entangle_cz_then_cx(0, 0, reg_wires[0], reg_wires[1], cz=False)

            ancilla_count = 1
            for r in range(self.registers):
                w = register_wires(r)
                anc = ancilla_count - 1
                idx = next(counter)
                schedule.append(('cz', anc, w[0], idx))
                if ancilla_count < self.ancilla:
                    ancilla_count += 1

        # same "last iteration swaps source/target" shape as `U1Circuit`'s ancilla-entangle.
        if self.registers > 1 and self.ancilla > 1:
            anc_wires = list(range(self.ancilla))
            if self.ancilla > 2:
                for i in range(self.ancilla):
                    if i != self.ancilla - 1:
                        entangle_cz_then_cx(i, i + 1, anc_wires, anc_wires, cz=False)
                    else:
                        entangle_cz_then_cx(0, i, anc_wires, anc_wires, cz=False)
            else:
                entangle_cz_then_cx(0, 1, anc_wires, anc_wires, cz=False)

        n_params = max((op[3] for op in schedule), default=-1) + 1
        return schedule, n_params


# --------------------------------------------------------------------------- #
# Control baselines: one independent circuit per channel, classical combination
# --------------------------------------------------------------------------- #
class QU1Control(nn.Module):
    """`circuits.py::Q_U1_control`. A single fixed 4-qubit circuit (rx embedding, then a `CXPowGate`
    ring 0->1->2->3->0), `PauliZ`-read on qubit 0, run **independently once per channel** with a
    private parameter set (`kernel` has an explicit channel axis), combined classically via a sum
    (or, with `classical_weights=True`, a learned per-position-per-channel affine reweight then
    sum -- the paper's "WEV" variant).
    """

    def __init__(self, n_kernels, n_input_channels, classical_weights=False,
                num_x=None, num_y=None):
        super().__init__()
        self.n_kernels = n_kernels
        self.n_input_channels = n_input_channels
        self.classical_weights = classical_weights
        self.n_qubits = N_FILTER_QUBITS

        # `Q_entangle(source, target, qubits)` gates `(qubits[source], qubits[target])` =
        # (control, rotated) -- note this is the OPPOSITE argument order from `Q_new_entangle`
        # (used by U1Circuit/U2Circuit above), which gates `(qubits_tar[target], qubits_src[source])`.
        # Ring: (0,1), (1,2), ..., (n-2,n-1), then the last edge is (0, n-1) -- control=0 again,
        # not a continuation of the ring -- matching `Q_entangle(0, i, ...)` for i = n_qubits-1.
        schedule = []
        for i in range(self.n_qubits):
            if i != self.n_qubits - 1:
                schedule.append((i, i + 1))
            else:
                schedule.append((0, i))
        self.gate_schedule = schedule
        self.n_params = len(schedule)

        self.kernel = nn.Parameter(_glorot_normal((n_kernels, n_input_channels, self.n_params)))
        if classical_weights:
            assert num_x is not None and num_y is not None
            self.channel_weights = nn.Parameter(
                torch.randn(num_x, num_y, n_input_channels, dtype=torch.float64) * 0.1 + 1.0)
            self.channel_bias = nn.Parameter(
                torch.randn(num_x, num_y, n_input_channels, dtype=torch.float64) * 0.1)

    def _circuit_unitary(self, params):
        n = self.n_qubits
        U = torch.eye(2 ** n, dtype=torch.complex128, device=params.device)
        for pidx, (control, target) in enumerate(self.gate_schedule):
            U = cx_pow(params[pidx], control, target, n) @ U
        return U

    def forward(self, images):
        patches = extract_patches(images)   # (B, num_x, num_y, C, 2, 2)
        b, num_x, num_y, c = patches.shape[:4]
        pixels = patches.reshape(b, num_x, num_y, c, N_FILTER_QUBITS)

        device = images.device
        n = self.n_qubits
        dim = 2 ** n

        outputs = []
        for k in range(self.n_kernels):
            per_channel = []
            for ch in range(self.n_input_channels):
                U = self._circuit_unitary(self.kernel[k, ch])
                angles = np.pi * pixels[:, :, :, ch, :].reshape(-1, self.n_qubits)
                state = _rx_state(angles, self.n_qubits, device)
                state = (U.unsqueeze(0) @ state.unsqueeze(-1)).squeeze(-1)
                expval = _pauli_op_expval(state, 0, pauli='Z', n_qubits=n)
                per_channel.append(expval.reshape(b, num_x, num_y))
            channel_out = torch.stack(per_channel, dim=-1)   # (B, num_x, num_y, C)
            if self.classical_weights:
                channel_out = channel_out * self.channel_weights + self.channel_bias
            outputs.append(channel_out.sum(dim=-1))
        stacked = torch.stack(outputs, dim=-1)   # (B, num_x, num_y, n_kernels)
        return un_embed(stacked)


class QU2Control(nn.Module):
    """`circuits.py::Q_U2_control`. Same "one independent circuit per channel, classical
    combination" structure as `QU1Control`, but the per-channel circuit is a literal QCNN-style
    layer over a 2x2 `GridQubit` grid: `CZPowGate` then `CXPowGate` between pairs at increasing
    step sizes `1, 2` (for a 4-qubit filter). `depth` plays the same role as `n_kernels`.
    """

    def __init__(self, depth, n_input_channels, classical_weights=False, num_x=None, num_y=None):
        super().__init__()
        self.depth = depth
        self.n_input_channels = n_input_channels
        self.classical_weights = classical_weights
        self.n_qubits = N_FILTER_QUBITS

        schedule = []
        step_sizes = [2 ** i for i in range(int(np.log2(self.n_qubits)))]
        for step in step_sizes:
            for target in range(0, self.n_qubits, 2 * step):
                schedule.append((target, target + step))   # (CZPow, CXPow) pair, same qubits
        self.gate_schedule = schedule
        self.n_params = 2 * len(schedule)

        self.kernel = nn.Parameter(_glorot_normal((depth, n_input_channels, self.n_params)))
        if classical_weights:
            assert num_x is not None and num_y is not None
            self.channel_weights = nn.Parameter(
                torch.randn(num_x, num_y, n_input_channels, dtype=torch.float64) * 0.1 + 1.0)
            self.channel_bias = nn.Parameter(
                torch.randn(num_x, num_y, n_input_channels, dtype=torch.float64) * 0.1)

    def _circuit_unitary(self, params):
        n = self.n_qubits
        U = torch.eye(2 ** n, dtype=torch.complex128, device=params.device)
        pidx = 0
        for target, other in self.gate_schedule:
            U = cz_pow(params[pidx], target, other, n) @ U
            pidx += 1
            U = cx_pow(params[pidx], target, other, n) @ U
            pidx += 1
        return U

    def forward(self, images):
        patches = extract_patches(images)
        b, num_x, num_y, c = patches.shape[:4]
        pixels = patches.reshape(b, num_x, num_y, c, N_FILTER_QUBITS)

        device = images.device
        n = self.n_qubits

        outputs = []
        for k in range(self.depth):
            per_channel = []
            for ch in range(self.n_input_channels):
                U = self._circuit_unitary(self.kernel[k, ch])
                angles = np.pi * pixels[:, :, :, ch, :].reshape(-1, self.n_qubits)
                state = _rx_state(angles, self.n_qubits, device)
                state = (U.unsqueeze(0) @ state.unsqueeze(-1)).squeeze(-1)
                expval = _pauli_op_expval(state, 0, pauli='Z', n_qubits=n)
                per_channel.append(expval.reshape(b, num_x, num_y))
            channel_out = torch.stack(per_channel, dim=-1)
            if self.classical_weights:
                channel_out = channel_out * self.channel_weights + self.channel_bias
            outputs.append(channel_out.sum(dim=-1))
        stacked = torch.stack(outputs, dim=-1)
        return un_embed(stacked)


def _rx_state(angles, n_qubits, device):
    """angles: (N, n_qubits) -> (N, 2^n_qubits) statevector from `rx(angles[:,i])` on wire i of
    |0...0>, batched per-wire (same trick as `_rx_embed_batch`, starting from the all-zero state).
    """
    n_rows = angles.shape[0]
    dim = 2 ** n_qubits
    state = torch.zeros(n_rows, dim, dtype=torch.complex128, device=device)
    state[:, 0] = 1.0
    return _rx_embed_batch(state, angles, list(range(n_qubits)), n_qubits)


# --------------------------------------------------------------------------- #
# Full model: quantum-conv layer -> Flatten -> Dense(32, relu) -> Dense(n_classes, softmax)
# --------------------------------------------------------------------------- #
class MCQCNNModel(nn.Module):
    """`models.py`'s model-assembly functions: one quantum-conv layer (any of the four circuit
    classes above) -> `Flatten` -> `Dense(32, relu)` -> `Dense(n_classes)` (softmax folded into
    `nn.CrossEntropyLoss` instead of an explicit final activation, matching every other classical
    head in this codebase).
    """

    def __init__(self, qconv, num_x, num_y, n_kernels, n_classes, relu=True):
        super().__init__()
        self.qconv = qconv
        self.relu = nn.ReLU() if relu else nn.Identity()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_x * num_y * n_kernels, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, images):
        """images: (B, H, W, C) channel-last, in [0, 1] (matching the original's `tf` layout)."""
        x = self.qconv(images.to(torch.float64))
        x = self.relu(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x.to(self.fc1.weight.dtype)))
        return self.fc2(x)


def conv_output_size(image_size, filter_size=FILTER_SIZE, stride=1):
    return (image_size - filter_size) // stride + 1


# --------------------------------------------------------------------------- #
# The ten trainable variants of `train.py`'s menu, one factory each
# --------------------------------------------------------------------------- #
# `models.py` ships eight assembly functions, but `train.py` offers ten menu entries: the two
# `PCO_*_QCNN_model` functions are each called twice, with `rdpa=3` (PCO) and `rdpa=1` (PCO-T).
# `rdpa` is "registers per ancilla" -- `ancilla = registers // rdpa` -- so with the fixed
# `registers=3` those two settings mean 1 shared ancilla vs. 3 (one per channel register).
#
# Every model in `models.py` hardcodes `n_kernels=3` (`depth=3` for the U2 control), which is the
# number of output feature maps of the quantum-conv layer.
N_KERNELS = 3


def _assemble(qconv, image_size, n_kernels, n_classes):
    num = conv_output_size(image_size)
    return MCQCNNModel(qconv, num_x=num, num_y=num, n_kernels=n_kernels, n_classes=n_classes)


def CO_U1_QCNN(image_size, n_input_channels, n_classes):
    """`models.py::CO_U1_QCNN_model` -- Channel Overwrite, U1 ansatz. One 4-qubit data register
    reused across channels (`registers=1`), so each channel's angle encoding *overwrites* the
    previous one's register state in a new `circuit_layer`; no inter-channel entangler.
    """
    qconv = U1Circuit(n_kernels=N_KERNELS, n_input_channels=n_input_channels)
    return _assemble(qconv, image_size, N_KERNELS, n_classes)


def PCO_U1_QCNN(image_size, n_input_channels, n_classes):
    """`models.py::PCO_U1_QCNN_model(rdpa=3)` -- Partial Channel Overwrite, U1 ansatz. Three
    parallel data registers (one per channel) entangled with each other in-circuit (`inter_U`),
    depositing phase on a single shared ancilla.
    """
    qconv = U1Circuit(n_kernels=N_KERNELS, n_input_channels=n_input_channels,
                      registers=3, rdpa=3, inter_U=True)
    return _assemble(qconv, image_size, N_KERNELS, n_classes)


def PCO_T_U1_QCNN(image_size, n_input_channels, n_classes):
    """`models.py::PCO_U1_QCNN_model(rdpa=1)` -- the "tertiary" PCO, U1 ansatz. Same as `PCO_U1`
    but with one ancilla *per* register (3 instead of 1), so the readout qubit is not a bottleneck.
    """
    qconv = U1Circuit(n_kernels=N_KERNELS, n_input_channels=n_input_channels,
                      registers=3, rdpa=1, inter_U=True)
    return _assemble(qconv, image_size, N_KERNELS, n_classes)


def WEV_U1_QCNN(image_size, n_input_channels, n_classes):
    """`models.py::QCNN_U1_weighted_control_model` -- Weighted Expectation Value, U1 ansatz. The
    control circuit run once per channel, but combined with a *learned* per-position affine
    reweight before the sum, instead of a plain sum.
    """
    num = conv_output_size(image_size)
    qconv = QU1Control(n_kernels=N_KERNELS, n_input_channels=n_input_channels,
                       classical_weights=True, num_x=num, num_y=num)
    return _assemble(qconv, image_size, N_KERNELS, n_classes)


def CONTROL_U1_QCNN(image_size, n_input_channels, n_classes):
    """`models.py::QCNN_U1_control_model` -- the paper's non-multi-channel-aware baseline, U1
    ansatz. A 4-qubit circuit run independently per channel with private parameters, combined by a
    plain classical sum. This is "an existing QCNN" in the paper's comparisons.
    """
    qconv = QU1Control(n_kernels=N_KERNELS, n_input_channels=n_input_channels,
                       classical_weights=False)
    return _assemble(qconv, image_size, N_KERNELS, n_classes)


def CO_U2_QCNN(image_size, n_input_channels, n_classes):
    """`models.py::CO_U2_QCNN_model` -- Channel Overwrite, U2 ansatz (each `CXPowGate` of U1 is
    joined by a `CZPowGate` on the same pair, doubling the parameters per entangling step).
    """
    qconv = U2Circuit(n_kernels=N_KERNELS, n_input_channels=n_input_channels)
    return _assemble(qconv, image_size, N_KERNELS, n_classes)


def PCO_U2_QCNN(image_size, n_input_channels, n_classes):
    """`models.py::PCO_U2_QCNN_model(rdpa=3)` -- Partial Channel Overwrite, U2 ansatz."""
    qconv = U2Circuit(n_kernels=N_KERNELS, n_input_channels=n_input_channels,
                      registers=3, rdpa=3, inter_U=True)
    return _assemble(qconv, image_size, N_KERNELS, n_classes)


def PCO_T_U2_QCNN(image_size, n_input_channels, n_classes):
    """`models.py::PCO_U2_QCNN_model(rdpa=1)` -- the "tertiary" PCO, U2 ansatz."""
    qconv = U2Circuit(n_kernels=N_KERNELS, n_input_channels=n_input_channels,
                      registers=3, rdpa=1, inter_U=True)
    return _assemble(qconv, image_size, N_KERNELS, n_classes)


def WEV_U2_QCNN(image_size, n_input_channels, n_classes):
    """`models.py::QCNN_U2_weighted_control_model` -- Weighted Expectation Value, U2 ansatz."""
    num = conv_output_size(image_size)
    qconv = QU2Control(depth=N_KERNELS, n_input_channels=n_input_channels,
                       classical_weights=True, num_x=num, num_y=num)
    return _assemble(qconv, image_size, N_KERNELS, n_classes)


def CONTROL_U2_QCNN(image_size, n_input_channels, n_classes):
    """`models.py::QCNN_U2_control_model` -- the non-multi-channel-aware baseline, U2 ansatz."""
    qconv = QU2Control(depth=N_KERNELS, n_input_channels=n_input_channels,
                       classical_weights=False)
    return _assemble(qconv, image_size, N_KERNELS, n_classes)


#: `train.py`'s menu order, so a notebook can select a variant by name.
VARIANTS = {
    'CO_U1': CO_U1_QCNN,
    'PCO_U1': PCO_U1_QCNN,
    'PCO_T_U1': PCO_T_U1_QCNN,
    'WEV_U1': WEV_U1_QCNN,
    'CONTROL_U1': CONTROL_U1_QCNN,
    'CO_U2': CO_U2_QCNN,
    'PCO_U2': PCO_U2_QCNN,
    'PCO_T_U2': PCO_T_U2_QCNN,
    'WEV_U2': WEV_U2_QCNN,
    'CONTROL_U2': CONTROL_U2_QCNN,
}

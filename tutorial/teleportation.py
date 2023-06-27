#!/usr/bin/env python3

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.quantum_info.states import state_fidelity
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
from math import pi
import numpy as np
import os


def state_n_qubit(state, id: int):
    rho = DensityMatrix(state)
    num = rho.num_qubits
    assert num is not None
    assert num > id
    pauli_singles = PauliList(["X", "Y", "Z"])
    for i in range(num):
        if num > 1:
            paulis = PauliList.from_symplectic(
                np.zeros((3, (num - 1)), dtype=bool),
                np.zeros((3, (num - 1)), dtype=bool),
            ).insert(i, pauli_singles, qubit=True)
        else:
            paulis = pauli_singles
        bloch_state = [
            np.real(np.trace(np.dot(mat, rho.data))) for mat in paulis.matrix_iter()
        ]
        if id == i:
            return bloch_state


def reference(simulator, theta: float, phi: float):
    qr = QuantumRegister(1)
    qc = QuantumCircuit(qr)

    qc.rz(phi, 0)
    qc.rx(theta, 0)
    res = execute(qc, simulator).result().get_statevector(qc)
    return res


# Initialize circuit and registers
qr = QuantumRegister(3)
crz = ClassicalRegister(1)
crx = ClassicalRegister(1)
qc = QuantumCircuit(qr, crz, crx)

# Initialize qbit 0 to be teleported to random state
theta = pi / 3
phi = pi / 4
if os.getenv("THETA") is not None:
    theta = float(os.getenv("THETA")) / 180.0 * pi
if os.getenv("PHI") is not None:
    phi = float(os.getenv("PHI")) / 180.0 * pi
print("theta = {}, phi = {}".format(theta, phi))
qc.rz(phi, 0)
qc.rx(theta, 0)

# create Bell pair on qbits 1 and 2
qc.barrier()
qc.h(1)
qc.cx(1, 2)

# Alice steps
qc.barrier()
qc.cx(0, 1)
qc.h(0)

qc.barrier()
qc.measure(0, 0)
qc.measure(1, 1)

# Bob steps
qc.barrier()
qc.z(2).c_if(crz, 1)
qc.x(2).c_if(crx, 1)
print(qc.draw(output="text"))

simulator = Aer.get_backend("statevector_simulator")
actual = state_n_qubit(execute(qc, simulator).result().get_statevector(qc), 2)
expected = state_n_qubit(reference(simulator, theta, phi), 0)
print("actual:   ", actual)
print("expected: ", expected)
if os.getenv("FIDELITY") is not None:
    print("fidelity: ", state_fidelity(actual, expected))

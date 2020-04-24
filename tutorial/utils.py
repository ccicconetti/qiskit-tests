"""Some utilities/wrappers for Qiskit"""

import numpy as np
from numpy import pi

from qiskit.providers.aer.noise import NoiseModel
from qiskit import(
    execute,
    Aer,
    IBMQ,
    QuantumCircuit
)

from qiskit.quantum_info import Pauli


class NoiseModelWrapper():
    "Load noise model from IBMQ real quantum computer"

    def __init__(self, ibmq_backend):
        print("Building circuit with noise from '{}'".format(ibmq_backend))

        # Build noise model from backend properties
        provider = IBMQ.load_account()
        backend = provider.get_backend(ibmq_backend)
        self.noise_model = NoiseModel.from_backend(backend)

        # Get coupling map from backend
        self.coupling_map = backend.configuration().coupling_map

        # Get basis gates from noise model
        self.basis_gates = self.noise_model.basis_gates

    def execute(self, qc):
        "Execute simulation with noise"

        result = execute(
            qc,
            Aer.get_backend('qasm_simulator'),
            coupling_map=self.coupling_map,
            basis_gates=self.basis_gates,
            noise_model=self.noise_model).result()
        return result

def bloch_states(rho):
    """Return the values of the Bloch vectors for a given state.

    Taken from plot_bloch_multivector in
    qiskit.visualization.state_visualization.
    """

    if rho.ndim == 1:
        rho = np.outer(rho, np.conj(rho))

    num = int(np.log2(len(rho)))

    ret = []
    for i in range(num):
        pauli_singles = [
            Pauli.pauli_single(num, i, 'X'),
            Pauli.pauli_single(num, i, 'Y'),
            Pauli.pauli_single(num, i, 'Z')
        ]

        ret.append(list(
            map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), rho))),
                pauli_singles)))

    return ret

def qft(N, print_circuit):
    """Return a circuit to compute the QFT for N qbits."""

    qft_circuit = QuantumCircuit(N, name="QFT")
    if N == 3:
        # Handle qbit 2
        qft_circuit.h(2)
        qft_circuit.cu1(pi/2, 1, 2)
        qft_circuit.cu1(pi/4, 0, 2)

        # Handle qbit 1
        qft_circuit.h(1)
        qft_circuit.cu1(pi/2, 0, 1)

        # Handle qqit 0
        qft_circuit.h(0)

        # Swap qbits 0 and 2
        qft_circuit.swap(0, 2)
    else:
        raise Exception("QFT only implemented with 3 qbits")
    
    if print_circuit:
        print(qft_circuit.draw(output='text'))

    return qft_circuit
    
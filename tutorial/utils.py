"""Some utilities/wrappers for Qiskit"""

import numpy as np
from numpy import pi
import operator

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

def decode_message(result, print_message=False):
    """Find an encoded message from count of probabilities."""

    message_decoded = max(
        result.get_counts().items(),
        key=operator.itemgetter(1))[0]

    tot_counts = sum(result.get_counts().values())

    if print_message:
        print("received message {} (prob {:.2f}%)".format(
            message_decoded,
            100 * float(result.get_counts()[message_decoded]) / tot_counts
        ))

    return message_decoded

def qft(N, print_circuit=False):
    """Return a circuit to compute the QFT for N qbits."""

    assert N > 0

    qft_circuit = QuantumCircuit(N, name="QFT")
    if N == 3:
        print("Special value N=3 for QFT circuit creation")

        # Handle qbit 2
        qft_circuit.h(2)
        qft_circuit.cu1(pi/2, 1, 2)
        qft_circuit.cu1(pi/4, 0, 2)

        # Handle qbit 1
        qft_circuit.h(1)
        qft_circuit.cu1(pi/2, 0, 1)

        # Handle qbit 0
        qft_circuit.h(0)

        # Swap qbits 0 and 2
        qft_circuit.swap(0, 2)
    else:
        # Add Hadamard and rotation gates
        for i in range(N):
            cur = N - i - 1
            qft_circuit.h(cur)
            for j in range(cur):
                qft_circuit.cu1(pi/2**(cur-j), j, cur)

        # Swap qbits
        for i in range(int(N/2)):
            qft_circuit.swap(i, N-1-i)

    if print_circuit:
        print(qft_circuit.draw(output='text'))

    return qft_circuit
    
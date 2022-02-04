"""Some utilities/wrappers for Qiskit"""

from ast import literal_eval
import json
import operator
from os import path

import numpy as np
from numpy import pi

from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap
from qiskit.tools.monitor import job_monitor
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

    def __init__(self, ibmq_backend, no_save_json=False, quiet=False):
        """Load a noise model from either a local file or IBMQ"""
        if not quiet:
            print("Building circuit with noise from '{}'".format(
                ibmq_backend))

        # If a file exists called like the backend (with a .json
        # extension), then load the model from that.
        # In this case, we also try to load a coupling map from
        # a file with .map extension. If it does not exist, no
        # worries, we just assume it is default (i.e., empty).
        json_filename = '{}.json'.format(ibmq_backend)
        coupling_map_filename = '{}.map'.format(ibmq_backend)
        if path.exists(json_filename):
            if not quiet:
                print('Loading noise model from {}'.format(json_filename))
            with open(json_filename, 'r') as infile:
                noise_model_dict = json.load(infile)
                self.noise_model = NoiseModel.from_dict(noise_model_dict)

            self.coupling_map = None
            if path.exists(coupling_map_filename):
                if not quiet:
                    print('Loading coupling map from {}'.format(coupling_map_filename))
                with open(coupling_map_filename, 'r') as coupling_infile:
                    self.coupling_map = CouplingMap(
                        literal_eval(coupling_infile.read()))

        # Otherwise, load the noise model from IBMQ (requires)
        # account properties to be stored in default location
        # and save the noise model in a JSON file for future use,
        # unless the no_save_json flag is set
        else:
            # Build noise model from backend properties
            provider = IBMQ.load_account()
            backend = provider.get_backend(ibmq_backend)
            self.noise_model = NoiseModel.from_backend(backend)

            # Get coupling map from backend
            self.coupling_map = backend.configuration().coupling_map

            # Save the model and coupling map (if not default) to file
            if not no_save_json:
                if not quiet:
                    print('Saving to {} the noise model for future use'.format(
                        json_filename))
                with open(json_filename, 'w') as outfile:
                    json.dump(self.noise_model.to_dict(True), outfile)
                if self.coupling_map is not None:
                    if not quiet:
                        print('Saving to {} the coupling map for future use'.format(
                            coupling_map_filename))
                    with open(coupling_map_filename, 'w') as coupling_outfile:
                        coupling_outfile.write(str(self.coupling_map))

    def execute(self, qc, shots=1024):
        "Execute simulation with noise"

        result = execute(
            qc,
            Aer.get_backend('qasm_simulator'),
            coupling_map=self.coupling_map,
            basis_gates=self.noise_model.basis_gates,
            noise_model=self.noise_model,
            shots=shots).result()
        return result

class IbmqWrapper():
    "Wrapper to execute circuit on an IBMQ real quantum computer"

    def __init__(self, ibmq_backend, quiet=False):
        self.quiet = quiet

        if not quiet:
            print("Loading IBMQ backend '{}'".format(
                ibmq_backend))

        # load IBM account
        self.provider = IBMQ.load_account()
        self.backend = self.provider.get_backend(ibmq_backend)

    def execute(self, qc, shots=1024):
        "Execute simulation"

        qc_compiled = transpile(
            qc,
            backend=self.backend,
            optimization_level=1)
        if not self.quiet:
            print(qc_compiled.draw(output='text'))

        job = execute(
            qc_compiled,
            backend=self.backend,
            shots=shots)
        job_monitor(job)
        result = job.result()
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

# From Qiskit Textbook
def qft_dagger(circ, n):
    """n-qubit QFTdagger the first n qubits in circ"""

    for qubit in range(n//2):
        circ.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            circ.cu1(-pi/float(2**(j-m)), m, j)
        circ.h(j)
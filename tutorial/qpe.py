"""Quantum phase estimation example inspired from the Qiskit Textbook"""

from random import uniform
# import matplotlib.pyplot as plt
from numpy import pi

from qiskit import (
    QuantumCircuit,
    ClassicalRegister,
    QuantumRegister,
    execute,
    BasicAer
)

# from qiskit.visualization import plot_bloch_multivector

from utils import (
    qft,
    qft_dagger,
    NoiseModelWrapper,
    decode_message
)

class QpeRotOperator():
    """Rotation operator (U1 gate) for which we estimate the phase theta"""

    def __init__(self, phase_qbit_index, theta):
        """Initialization"""

        self.ndx = phase_qbit_index
        self.theta = theta

    def prepareInput(self, quantum_circuit):
        """Input preparation"""

        # For the U1 gate:
        #   |psi> = |1>
        # is an eigenvector, with eigenvalue
        # e^(pi * theta)
        quantum_circuit.x(self.ndx)

    def applyOperator(self, quantum_circuit, control_qbit_index):
        """Apply cU1"""

        quantum_circuit.cu1(
            2 * self.theta * pi,
            control_qbit_index,
            self.ndx)

class QpeYOperator():
    """Y operator for which we estimate the phase"""

    def __init__(self, phase_qbit_index):
        """Initialization"""

        self.ndx = phase_qbit_index

    def prepareInput(self, quantum_circuit):
        """Input preparation"""

        #
        # For the Y gate:
        # 
        #   |psi> = 1 / sqrt(2) |0> + i / sqrt(2) |1>
        #
        # is an eigenvector, with eigen values +-1
        #
        quantum_circuit.h(self.ndx)
        quantum_circuit.s(self.ndx)

    def applyOperator(self, quantum_circuit, control_qbit_index):
        """Apply cY"""

        quantum_circuit.cy(
            control_qbit_index,
            self.ndx)

#
# Configuration
#

N = 4 # number of qbits for QPE
experiment_type = 'simulator-qasm'
shots = 1024
gate_under_test = 'Y'
theta = uniform(-1, 1) # unused with Y gate
qpe_operator = QpeRotOperator(N, theta) \
    if gate_under_test == 'U1' \
    else QpeYOperator(N)

# True: use function from Qiskit Textbook
# False: create QFT then invert it
use_qft_dagger = True

#
# Execution
#

assert gate_under_test in ['U1', 'Y']

# Load noisy gates, if required
noise_wrapper = None
if experiment_type == 'simulator-noise':
    noise_wrapper = NoiseModelWrapper('ibmq_essex')

# Initialize circuit and registers
qr = QuantumRegister(N+1) # N ancillas + 1 phase estimation
cr = ClassicalRegister(N)
qc = QuantumCircuit(qr, cr)

# Prepare ancilla qbits
for i in range(N):
    qc.h(i)

# Prepare phase estimation qbit
qpe_operator.prepareInput(qc)

# Perform cU operations
repetitions = 1
for ancilla_qbit in range(N):
    for i in range(repetitions):
        qpe_operator.applyOperator(qc, ancilla_qbit)
    repetitions *= 2

# Add barrier before QFT^-1
qc.barrier()

# Add QFT^-1 to the ancilla bits
if use_qft_dagger:
    qft_dagger(qc, N)
else:
    qc.append(qft(N, print_circuit=True).inverse(), qr[0:N])

# Measurement
qc.barrier()
for i in range(N):
    qc.measure(i, i)

# Print circuit
print(qc.draw(output='text'))

# Execute circuit
if experiment_type == 'simulator-qasm':
    backend = BasicAer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=shots).result()

elif experiment_type == 'simulator-noise':
    assert noise_wrapper is not None
    result = noise_wrapper.execute(qc, shots=shots)

# Find measurement with maximum probability
print(result.get_counts())
estimated_theta = \
    int(decode_message(result, print_message=True), 2) / 2**N

if gate_under_test == 'U1':
    if theta < 0:
        estimated_theta -= 1
    print('U1-gate: phase expected {:.3f}, estimated {:.3f} (error = {:.2f}%)'.format(
        theta,
        estimated_theta,
        100 * abs(theta - estimated_theta) / theta))
else:
    print('Y-gate: expected 0, found {}'.format(estimated_theta))
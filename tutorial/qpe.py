"""Quantum phase estimation example inspired from the Qiskit Textbook"""

import argparse
import pickle
from random import uniform

# import matplotlib.pyplot as plt
from numpy import pi

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, BasicAer
from qiskit.result import Result

# from qiskit.visualization import plot_bloch_multivector

from utils import qft, qft_dagger, NoiseModelWrapper, decode_message


class QpeRotOperator:
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

        quantum_circuit.cu1(2 * self.theta * pi, control_qbit_index, self.ndx)


class QpeYOperator:
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

        quantum_circuit.cy(control_qbit_index, self.ndx)


#
# Command-line arguments
#

parser = argparse.ArgumentParser(
    "Quantum phase estimation experiment",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--output", type=str, default="", help="Output file with results")
parser.add_argument(
    "--load", type=str, default="", help="Do not run experiment, load results from file"
)
parser.add_argument(
    "--experiment_type",
    type=str,
    default="simulator-qasm",
    help="One of: simulator-qasm, simulator-noise, or real",
)
parser.add_argument(
    "--backend",
    type=str,
    default="",
    help="IBMQ backend that runs the experiment or used as import the noise model",
)
parser.add_argument(
    "--project", type=str, default="", help="Indicate a specific hub,group,project"
)
parser.add_argument("--shots", type=int, default=1024, help="Number of shots to run")
parser.add_argument(
    "--num_qubits", type=int, default=4, help="Number of qubits for QPE"
)
parser.add_argument("--gate_under_test", type=str, default="Y", help="One of: U1, Y")
parser.add_argument(
    "--use_qft_dagger",
    action="store_true",
    default=False,
    help="Use function from Qiskit textbook instead of creating QFT and inverting it",
)
parser.add_argument(
    "--print_circuit",
    action="store_true",
    default=False,
    help="Print the circuit",
)
args = parser.parse_args()

#
# Configuration and initial parameter checks
#

assert args.num_qubits > 0
assert args.gate_under_test in ["U1", "Y"]
assert args.experiment_type in ["simulator-qasm", "simulator-noise", "real"]
if args.experiment_type in ["simulator-noise", "real"]:
    if args.backend == "":
        raise RuntimeError("Must specify a backend with simulator-noise or real")

N = args.num_qubits
theta = uniform(-1, 1)  # unused with Y gate
qpe_operator = (
    QpeRotOperator(N, theta) if args.gate_under_test == "U1" else QpeYOperator(N)
)


#
# Execution
#

if args.load == "":
    # Load noisy gates, if required
    noise_wrapper = None
    if args.experiment_type == "simulator-noise":
        noise_wrapper = NoiseModelWrapper("ibmq_bogota", no_save=True)

    # Initialize circuit and registers
    qr = QuantumRegister(N + 1)  # N ancillas + 1 phase estimation
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
    if args.use_qft_dagger:
        qft_dagger(qc, N)
    else:
        qc.append(qft(N, print_circuit=args.print_circuit).inverse(), qr[0:N])

    # Measurement
    qc.barrier()
    for i in range(N):
        qc.measure(i, i)

    # Print circuit
    if args.print_circuit:
        print(qc.draw(output="text"))

    # Execute circuit
    if args.experiment_type == "simulator-qasm":
        backend = BasicAer.get_backend("qasm_simulator")
        result = execute(qc, backend, shots=args.shots).result()

    elif args.experiment_type == "simulator-noise":
        assert noise_wrapper is not None
        result = noise_wrapper.execute(qc, shots=args.shots)

    if args.output != "":
        with open(args.output, "wb") as outfile:
            pickle.dump(result.to_dict(), outfile)

else:
    # Load circuit
    with open(args.load, "rb") as infile:
        result = Result.from_dict(pickle.load(infile))

        # Find measurement with maximum probability
        print(result.get_counts())
        estimated_theta = int(decode_message(result, print_message=True), 2) / 2**N

        if args.gate_under_test == "U1":
            if theta < 0:
                estimated_theta -= 1
            print(
                "U1-gate: phase expected {:.3f}, estimated {:.3f} (error = {:.2f}%)".format(
                    theta, estimated_theta, 100 * abs(theta - estimated_theta) / theta
                )
            )
        else:
            print("Y-gate: expected 0, found {}".format(estimated_theta))

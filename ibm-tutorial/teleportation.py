"""Teleportation example inspired from the Qiskit Textbook"""

import math
import random
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer
from qiskit.visualization import plot_bloch_multivector

from utils import NoiseModelWrapper

#
# Configuration
#

experiment_type = "simulator-statevector"
shots = 1024
bit_to_send = random.choice(["0", "1"])

#
# Execution
#

# Load noisy gates, if required
noise_wrapper = None
if experiment_type == "simulator-noise":
    noise_wrapper = NoiseModelWrapper("ibmq_manila")

# Initialize circuit and registers
qr = QuantumRegister(3)  # Protocol uses 3 qubits
crz = ClassicalRegister(1)  # and 2 classical bits
crx = ClassicalRegister(1)  # in 2 different registers
qc = QuantumCircuit(qr, crz, crx)

# Initialize qbit 0 to be teleported to random state
random.seed(0)
theta = random.uniform(0, math.pi)
phi = random.uniform(0, 2 * math.pi)
print("theta = {}, phi = {}".format(theta, phi))
qc.rz(phi, 0)
qc.rx(theta, 0)

# create Bell pair on qbits 1 and 2
qc.barrier()
qc.h(1)
qc.cx(1, 2)

# Decide which bit to teleport: |0> or |1>
if bit_to_send == "0":
    print("Sending a |0>")
    qc.id(0)
else:
    print("Sending a |1>")
    if bit_to_send != "1":
        raise Exception("Invalid bit_to_send value: {}".format(bit_to_send))
    qc.x(0)

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

#
# Test circuit
#
if experiment_type == "simulator-statevector":
    # Print circuit
    print(qc.draw(output="text"))

    # Run statevector simulation
    backend = BasicAer.get_backend("statevector_simulator")
    out_vector = execute(qc, backend).result().get_statevector(decimals=2)
    print(out_vector[2])
    plot_bloch_multivector(out_vector, title="Actual")

    if bit_to_send == "1":
        theta = math.pi - theta
        phi = 2 * math.pi - phi

    exp_vector = [
        complex(math.cos(theta / 2), 0),
        complex(
            math.cos(phi) * math.sin(theta / 2), math.sin(phi) * math.sin(theta / 2)
        ),
    ]
    plot_bloch_multivector(exp_vector, title="Expected")

    plt.show(block=True)

else:
    # Disentangle teleported qbit
    qc.rx(-theta, 2)
    qc.rz(-phi, 2)

    # Measure output
    crr = ClassicalRegister(1)
    qc.add_register(crr)
    qc.measure(2, 2)
    print(qc.draw(output="text"))

    if experiment_type == "simulator-qasm":
        backend = BasicAer.get_backend("qasm_simulator")
        result = execute(qc, backend, shots=shots).result()
        print(result.get_counts())
        count_0 = 0
        for k, v in result.get_counts().items():
            if k[0] == bit_to_send:
                count_0 += v
        if count_0 == shots:
            print("Teleportation successful")
        else:
            print("Teleportation failed")

    elif experiment_type == "simulator-noise":
        assert noise_wrapper is not None
        result = noise_wrapper.execute(qc)
        print(result.get_counts())

        count = {"0": 0, "1": 0}
        for k, v in result.get_counts().items():
            count[k[0]] += v
        print("Correct bit: {}%".format(int(round(count[bit_to_send] / shots * 100))))

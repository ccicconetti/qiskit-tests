"""Teleportation example from Qiskit Textbook"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer, IBMQ
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.extensions import Initialize

experiment_type = 'simulator-statevector'

qr = QuantumRegister(3)    # Protocol uses 3 qubits
crz = ClassicalRegister(1) # and 2 classical bits
crx = ClassicalRegister(1) # in 2 different registers
qc = QuantumCircuit(qr, crz, crx)

# initialize qbit 0 to be teleported to random state
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

# Print circuit
print(qc.draw(output='text'))

# Test circuit
if experiment_type == 'simulator-statevector':
    backend = BasicAer.get_backend('statevector_simulator')
    out_vector = execute(qc, backend).result().get_statevector(decimals=2)
    print(out_vector[2])
    plot_bloch_multivector(out_vector, title="Actual")

    exp_vector = [
        complex(math.cos(theta/2), 0),
        complex(math.cos(phi) * math.sin(theta/2), math.sin(phi) * math.sin(theta/2))
    ]
    plot_bloch_multivector(exp_vector, title="Expected")

    plt.show(block=True)
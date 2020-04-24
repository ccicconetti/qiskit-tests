"""QFT example inspired from the Qiskit Textbook"""

import operator
import math
import random
# import matplotlib.pyplot as plt
from numpy import pi

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    execute,
    BasicAer
)

# from qiskit.visualization import plot_bloch_multivector

from utils import (
    NoiseModelWrapper,
    bloch_states,
    qft
)

#
# Configuration
#

N = 3 # number of qbits
experiment_type = 'simulator-noise'
shots = 1024
num_encoded = random.randint(0, 2**N - 1)

#
# Execution
#

# Load noisy gates, if required
noise_wrapper = None
if experiment_type == 'simulator-noise':
    noise_wrapper = NoiseModelWrapper('ibmq_essex')

# Initialize circuit and registers
qr = QuantumRegister(N)    # Protocol needs Bell pair of qubits
qc = QuantumCircuit(qr)

# Prepare input based on the encoded number
print("Encoded message: {}".format(bin(num_encoded)))

if 'statevector' in experiment_type:
    #
    # With statevector simulation we encode the number in
    # the computational basis: for every binary position from
    # right to left (least significant to most signifcant) we
    # add a NOT gate only if it is 1. Below, we perform the QFT
    # 
    for i in range(N):
        if num_encoded & 2**i:
            qc.x(i)

else:
    #
    # For actual computations (also simulated) we encode the input
    # as the expected output after the QFT, then we perform the
    # QFT^-1 below.
    #
    for i in range(N):
        qc.h(i)
        qc.u1(num_encoded * 2 * pi / 2**(N-i), i)

# Add barrier before computation
qc.barrier()

# Take the QFT or QFT^-1 circuit depending on experiment type
if 'statevector' in experiment_type:
    qc.append(qft(N, print_circuit=True), qr)
else:
    qc.append(qft(N, print_circuit=True).inverse(), qr)
    qc.measure_all()

# Print circuit
print(qc.draw(output='text'))

#
# Test circuit
#
if experiment_type == 'simulator-statevector':
    # Run statevector simulation
    backend = BasicAer.get_backend('statevector_simulator')
    out_vector = execute(qc, backend).result().get_statevector()
    
    # Find the projects in the bloch sphere for all qbits
    projections = bloch_states(out_vector)

    # Check and print qbit by qbit in the computational basis
    for i in range(N):
        # Angle in the X-Y plane (Z should be 0)
        phi = math.atan2(projections[i][1], projections[i][0]) 
        # Shift phi in [0, 2*pi]
        phi = (2 * pi + phi) % (2 * pi)
        # Expected
        phi_expected = (num_encoded * 2 * pi / (2**(N-i))) % (2 * pi)

        print("{}-th qbit: projections {}, phi = {:.0f} {} {}".format(
            i,
            [round(x, 2) for x in projections[i]],
            phi*180/pi,
            'OK' if abs(phi - phi_expected) < 1e-5 else 'NOK',
            '(valid Z)' if projections[i][2] < 1e-5 else '(invalid Z)'
            ))
        
    # plot_bloch_multivector(out_vector)
    # plt.show(block=True)

else:
    if experiment_type == 'simulator-qasm':
        backend = BasicAer.get_backend('qasm_simulator')
        result = execute(qc, backend, shots=shots).result()

    elif experiment_type == 'simulator-noise':
        assert noise_wrapper is not None
        result = noise_wrapper.execute(qc)
    
    # Find measurement with maximum probability
    print(result.get_counts())

    message_decoded = max(
        result.get_counts().items(),
        key=operator.itemgetter(1))[0]
    print("received message {} (prob {}%)".format(
        message_decoded,
        100 * float(result.get_counts()[message_decoded]) / shots
    ))
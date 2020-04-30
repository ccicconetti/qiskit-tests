"""Simulate operations on one qubit"""

from math import pi, atan2, sqrt

import matplotlib.pyplot as plt

from qiskit import(
    QuantumCircuit,
    execute,
    BasicAer)
from utils import NoiseModelWrapper, IbmqWrapper

# Configuration
shots = 8*1024
steps = 19
angle_min = 0
angle_max = pi
verbose = False
print_circuit_and_exit = False
experiment_type = 'simulator-noise'

assert experiment_type in ['ibmq', 'simulator-noise', 'simulator-qasm']

# Load noisy gates, if required
backend_wrapper = None
if experiment_type == 'simulator-noise':
    backend_wrapper = NoiseModelWrapper('ibmq_armonk', quiet=not verbose)
elif experiment_type == 'ibmq':
    backend_wrapper = IbmqWrapper('ibmq_essex', quiet=not verbose)

# Run experiments
angle_span = angle_max - angle_min
angle_step = angle_span / (steps - 1)
if verbose:
    print("Iterating from {} to {} in steps of {} degrees".format(
        angle_min * 180 / pi,
        angle_max * 180 / pi,
        angle_step * 180 / pi
    ))

data = dict()
for i in range(steps):
    angle = angle_min + angle_step * i

    # Circuit preparation
    qc = QuantumCircuit(1)

    qc.rx(angle, 0)
    qc.measure_all()

    if verbose:
        print("#{} angle = {:.2f} degrees".format(i, angle * 180 / pi))
        print(qc.draw(output='text'))

    if print_circuit_and_exit:
        qc.draw(output='mpl')
        plt.show()
        break


    # Execution
    if experiment_type == 'simulator-qasm':
        result = execute(
            qc,
            BasicAer.get_backend('qasm_simulator'),
            shots=shots).result()

    else:
        assert backend_wrapper is not None
        result = backend_wrapper.execute(qc, shots=shots)

    # Recover angle from counts
    counts = result.get_counts()
    if '1' not in counts:
        est_angle = 0
    elif '0' not in counts:
        est_angle = pi
    else:
        est_angle = 2 * atan2(sqrt(counts['1']), sqrt(counts['0']))
        # est_angle = pi * counts['1'] / shots
    data[angle] = est_angle
    if verbose:
        print(counts)

for k, v in data.items():
    error = abs(k-v)
    print("{:.2f} {:.2f} (err {:.2f} degrees) {}".format(
        k * 180 / pi,
        v * 180 / pi,
        error * 180 / pi,
        'OK' if error < angle_step/2 else 'NOK'))
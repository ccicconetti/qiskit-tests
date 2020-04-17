"""Deutsch example"""

import random
from qiskit import(
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer)
#from qiskit.quantum_info import (
#    Statevector
#)
# import matplotlib.pyplot as plt
# from qiskit.tools.visualization import (
#   plot_state_city)

def blackbox_c(a_qc):
    """Constant function"""
    #a_qc.iden(0)
    if random.random() > 0.5:
        print("Constant function: returns always 0")
        a_qc.iden(1)
    else:
        print("Constant function: returns always 1")
        a_qc.iden(1)

def blackbox_b(a_qc):
    """Balanced function"""
    print("Balanced function")
    a_qc.cx(0, 1)

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')
#simulator = Aer.get_backend('statevector_simulator')

# Create the Quantum Circuit
q = QuantumRegister(2, name='q')
c = ClassicalRegister(1, name='c')
qc = QuantumCircuit(q, c, name='qc')

# Prepare input
qc.h(0)
qc.x(1)
qc.h(1)
qc.barrier()

# Feed input to Uf
blackbox_c(qc)

# Retrieve measurement on qbit#0
qc.barrier()
qc.h(0)
#qc.h(1)

# Performance measurement
qc.barrier()
qc.measure(q[0], c[0])

# Print circuit
print(qc.draw(output='text'))

# Print input
# print(Statevector.from_instruction(qc).data)

# Execute the circuit on the statevector simulator
job = execute(qc, simulator, shots=1000)

# Grab results from the job
result = job.result()

# Print result
# print(result.get_statevector(qc))
print(result.get_counts())

# plot_state_city(statevector, title='State vector')
# plt.show(block=True)
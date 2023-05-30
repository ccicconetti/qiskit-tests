"""Print the state of a qbit after a quantum gate"""

from math import pi
from qiskit import(
    QuantumCircuit,
    execute,
    Aer)
from utils import bloch_states

circuit = QuantumCircuit(1)

circuit.h(0)
circuit.s(0)

print(circuit.draw(output='text'))

out_vector = execute(
    circuit,
    Aer.get_backend('statevector_simulator'),
    shots=1).result().get_statevector()

print('state: {}'.format(out_vector))
print('Bloch: {}'.format(bloch_states(out_vector)[0]))
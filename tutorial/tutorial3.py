"""Tutorial"""

from qiskit import(
    QuantumCircuit,
    ClassicalRegister,
    QuantumRegister,
    Aer)

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

# Create a Quantum Circuit acting on the q register
q = QuantumRegister(3, name='goofy')
q_another = QuantumRegister(1, name='minnie')
c = ClassicalRegister(3, name='mickey')
circuit = QuantumCircuit(q, q_another, c, name='disney')

# Add a H gate on qubit 0
circuit.h(q[0])
circuit.iden(q[1])
circuit.iden(q[2])
circuit.measure(q[0], c[0])

inst = circuit.qasm()

print(inst[36:])

print(circuit.qregs)

print(circuit.cregs)

print(circuit.data)
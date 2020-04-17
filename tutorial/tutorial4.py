"""Tutorial"""

from qiskit import(
    QuantumCircuit,
    ClassicalRegister,
    QuantumRegister,
    execute,
    Aer)

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

# Create a Quantum Circuit acting on the q register
q = QuantumRegister(1, name='q')
c = ClassicalRegister(1, name='c')
qc = QuantumCircuit(q, c, name='qc')

qc.h(q[0])
qc.measure(q, c)

# execute the circuit
result = execute(qc, simulator).result()

qc.x(q[0]).c_if(c, 0)
qc.h(q[0]).c_if(c, 1)

print(qc.draw(output='text'))

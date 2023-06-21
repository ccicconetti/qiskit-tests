#!/usr/bin/env python3

from qiskit.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import SPSA

import argparse

seed = 0
iterations = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TwoLocal", description="Print Ansatz circuit for VQE"
    )
    parser.add_argument("-q", action="store_true", help="quiet mode")
    parser.add_argument("--num_qubits", type=int, help="number of qubits")
    parser.add_argument("--qasm", help="where to save the qasm file")
    parser.add_argument("--entanglement", default="full", help="entanglement strategy")
    parser.add_argument(
        "--rotation_blocks", default=["ry"], action="append", help="rotation blocks"
    )
    parser.add_argument(
        "--entanglement_blocks", default="cz", help="entanglement blocks"
    )
    args = parser.parse_args()

    # create a QUBO
    qubo = QuadraticProgram(name="QUBO test")
    qubo.binary_var_list(args.num_qubits)
    qubo.minimize(linear=[1] * args.num_qubits)
    # print(qubo.prettyprint())
    op, offset = qubo.to_ising()

    algorithm_globals.random_seed = seed
    noiseless_estimator = AerEstimator(
        run_options={"seed": seed, "shots": 1},
        transpile_options={"seed_transpiler": seed},
    )
    ansatz = TwoLocal(
        rotation_blocks=args.rotation_blocks,
        entanglement_blocks=args.entanglement_blocks,
        entanglement=args.entanglement,
    )
    spsa = SPSA(maxiter=iterations)
    vqe = VQE(noiseless_estimator, ansatz, optimizer=spsa)
    result_vqe = vqe.compute_minimum_eigenvalue(operator=op)
    if not args.q:
        print("VQE quantum circuit:")
        print(vqe.ansatz.decompose())
    if args.qasm:
        circ = vqe.ansatz.decompose()
        bound = circ.assign_parameters(list(range(circ.num_parameters)))
        with open(args.qasm, "w") as outfile:
            outfile.write(bound.qasm())

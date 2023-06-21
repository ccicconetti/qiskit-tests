#!/usr/bin/env python3

import numpy as np
import networkx as nx

from qiskit_aer import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo


import time
import argparse

seed = 42


def create_tsp():
    # Generating a graph of 3 nodes
    n = 3
    num_qubits = n**2
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_weighted_edges_from(
        [
            [0, 1, 10],
            [0, 2, 3],
            [0, 3, 5],
            [1, 3, 4],
            [2, 3, 1],
        ]
    )
    G.add_edge(1, 2, weight=1 + G.size(weight="weight"))
    tsp = Tsp(G)
    adj_matrix = nx.to_numpy_array(tsp.graph)
    if not args.q:
        print("distance\n", adj_matrix)

    return tsp, adj_matrix


def tsp_to_qubo(tsp: Tsp):
    qp = tsp.to_quadratic_program()
    if not args.q:
        print(qp.prettyprint())

    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(qp)
    qubitOp, offset = qubo.to_ising()
    if not args.q:
        print("Offset:", offset)
        print("Ising Hamiltonian:")
        print(str(qubitOp))

    return qubo, qubitOp, offset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TspExample", description="Example TSP solver with VQE"
    )
    parser.add_argument("-q", action="store_true", help="quiet mode")
    parser.add_argument("--mode", type=str, default="exact", help="run mode")
    # parser.add_argument("--qasm", help="where to save the qasm file")
    parser.add_argument("--entanglement", default="full", help="entanglement strategy")
    parser.add_argument(
        "--rotation_blocks", default=["ry"], action="append", help="rotation blocks"
    )
    parser.add_argument(
        "--entanglement_blocks", default="cz", help="entanglement blocks"
    )
    parser.add_argument(
        "--reps",
        default=5,
        help="number of repetitions of rotation/entanglement layers",
    )
    parser.add_argument("--iterations", default=125, help="number of iterations")
    args = parser.parse_args()
    assert args.mode in ["exact", "exact-hamiltonian", "vqe-ideal"]

    tsp, adj_matrix = create_tsp()
    qubo, qubitOp, offset = tsp_to_qubo(tsp)

    if args.mode == "exact":
        exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        result = exact.solve(qubo)
        if not args.q:
            print(result.prettyprint())

    else:
        if args.mode == "exact-hamiltonian":
            ee = NumPyMinimumEigensolver()
            start_time = time.time()
            result = ee.compute_minimum_eigenvalue(qubitOp)
            elapsed_time = time.time() - start_time

        elif args.mode == "vqe-ideal":
            algorithm_globals.random_seed = seed
            optimizer = SPSA(maxiter=args.iterations)
            ansatz = TwoLocal(
                qubitOp.num_qubits,
                args.rotation_blocks,
                args.entanglement_blocks,
                reps=args.reps,
                entanglement=args.entanglement,
            )
            vqe = SamplingVQE(sampler=Sampler(), ansatz=ansatz, optimizer=optimizer)
            result = vqe.compute_minimum_eigenvalue(qubitOp)
            elapsed_time = result.optimizer_time

        if not args.q:
            x = tsp.sample_most_likely(result.eigenstate)
            z = tsp.interpret(x)
            print("time:", elapsed_time)
            print("energy:", result.eigenvalue.real)
            print("tsp objective:", result.eigenvalue.real + offset)
            print("feasible:", qubo.is_feasible(x))
            print("solution:", z)
            print("solution objective:", tsp.tsp_value(z, adj_matrix))

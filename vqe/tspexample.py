#!/usr/bin/env python3

import numpy as np
import networkx as nx
import time
import argparse

from qiskit_aer import Aer
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Tsp
from qiskit.algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.utils import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.providers.fake_provider.utils.configurable_backend import (
    ConfigurableFakeBackend,
)
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Estimator


def create_tsp():
    # example network with four nodes
    if not args.random:
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

    else:
        tsp = Tsp.create_random_instance(args.random, seed=args.seed)

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
    modes = ["exact", "exact-hamiltonian", "vqe-ideal", "vqe-fake", "vqe-real"]
    parser = argparse.ArgumentParser(
        prog="TspExample",
        description="Example TSP solver with VQE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-q", action="store_true", help="quiet mode")
    parser.add_argument(
        "--seed", default=42, type=int, help="random number generator seed"
    )
    parser.add_argument(
        "--random",
        type=int,
        help="generate a random network with the given number of nodes, instead of the 4-node example network",
    )
    parser.add_argument(
        "--mode", type=str, default="exact", help=f"run mode, one of: {','.join(modes)}"
    )
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
    parser.add_argument(
        "--iterations", type=int, default=125, help="number of iterations"
    )
    parser.add_argument(
        "--intermediate", help="save the intermediate results to the specified file"
    )
    parser.add_argument("--backend", help="name of the IBMQ backend to use")
    args = parser.parse_args()

    # check arguments
    assert args.mode in modes
    assert args.mode != "vqe-real" or args.backend is not None

    # create the problem
    tsp, adj_matrix = create_tsp()
    qubo, qubitOp, offset = tsp_to_qubo(tsp)

    # solve the problem
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

            if not args.q:
                x = tsp.sample_most_likely(result.eigenstate)
                z = tsp.interpret(x)
                print("feasible:", qubo.is_feasible(x))
                print("solution:", z)
                print("solution objective:", tsp.tsp_value(z, adj_matrix))

        elif "vqe-" in args.mode:
            algorithm_globals.random_seed = args.seed
            optimizer = SPSA(maxiter=args.iterations)
            ansatz = TwoLocal(
                qubitOp.num_qubits,
                args.rotation_blocks,
                args.entanglement_blocks,
                reps=args.reps,
                entanglement=args.entanglement,
            )

            values = []

            def store_intermediate_result(eval_count, parameters, mean, std):
                values.append(mean.real)

            if args.mode == "vqe-ideal":
                estimator = AerEstimator(
                    run_options={"seed": args.seed, "shots": 1024},
                    transpile_options={"seed_transpiler": args.seed},
                )

            elif args.mode == "vqe-fake":
                device = ConfigurableFakeBackend(
                    "my-fake",
                    qubitOp.num_qubits,
                    version="0.0.1",
                    basis_gates=["id", "u1", "u2", "u3", "cx"],
                    qubit_t1=113.0,
                    qubit_t2=150.2,
                    qubit_frequency=4.8,
                    qubit_readout_error=0.04,
                    single_qubit_gates=["id", "u1", "u2", "u3"],
                    dt=1.33,
                )

                estimator = AerEstimator(
                    backend_options={
                        "method": "density_matrix",
                        "coupling_map": device.configuration().coupling_map,
                        "noise_model": NoiseModel.from_backend(device),
                    },
                    run_options={"seed": args.seed, "shots": 1024},
                    transpile_options={"seed_transpiler": args.seed},
                )

            elif args.mode == "vqe-real":
                service = QiskitRuntimeService()
                options = Options(optimization_level=1)
                options.execution.shots = 1024

                with Session(service=service, backend=args.backend) as session:
                    estimator = Estimator(session=session, options=options)

                    vqe = VQE(
                        estimator=estimator,
                        ansatz=ansatz,
                        optimizer=optimizer,
                        callback=store_intermediate_result,
                    )
                    result = vqe.compute_minimum_eigenvalue(qubitOp)
                    elapsed_time = result.optimizer_time

            if args.mode != "vqe-real":
                vqe = VQE(
                    estimator=estimator,
                    ansatz=ansatz,
                    optimizer=optimizer,
                    callback=store_intermediate_result,
                )
                result = vqe.compute_minimum_eigenvalue(qubitOp)
                elapsed_time = result.optimizer_time

            if not args.q:
                print("cost function evals:", result.cost_function_evals)
                print("optimal circuit:\n", result.optimal_circuit.decompose())
                print("optimal parameters: ", result.optimal_parameters)

        if not args.q:
            print("time:", elapsed_time)
            print("energy:", result.eigenvalue.real)
            print("tsp objective:", result.eigenvalue.real + offset)

    if args.intermediate:
        with open(args.intermediate, "w") as outfile:
            for v in values:
                outfile.write(f"{v}\n")

#!/usr/bin/env python3

from qiskit_ibm_runtime import QiskitRuntimeService

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="backends",
        description="Operation on IBM backends",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend", type=str, help="select this backend")
    args = parser.parse_args()

    service = QiskitRuntimeService()

    if args.backend is None:
        # Display all backends you have access.
        backends = service.backends()
        print("available backends:")
        for backend in backends:
            print(backend)

    else:
        # Print backend coupling map of the first backend
        backend = service.backend(args.backend)
        print("coupling map:\n", backend.coupling_map)

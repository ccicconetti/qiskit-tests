"""Some utilities/wrappers for Qiskit"""

from qiskit.providers.aer.noise import NoiseModel
from qiskit import(
    execute,
    Aer,
    IBMQ
)

class NoiseModelWrapper():
    "Load noise model from IBMQ real quantum computer"

    def __init__(self, ibmq_backend):
        print("Building circuit with noise from '{}'".format(ibmq_backend))

        # Build noise model from backend properties
        provider = IBMQ.load_account()
        backend = provider.get_backend(ibmq_backend)
        self.noise_model = NoiseModel.from_backend(backend)

        # Get coupling map from backend
        self.coupling_map = backend.configuration().coupling_map

        # Get basis gates from noise model
        self.basis_gates = self.noise_model.basis_gates

    def execute(self, qc):
        "Execute simulation with noise"

        result = execute(qc,
                    Aer.get_backend('qasm_simulator'),
                    coupling_map=self.coupling_map,
                    basis_gates=self.basis_gates,
                    noise_model=self.noise_model).result()
        return result

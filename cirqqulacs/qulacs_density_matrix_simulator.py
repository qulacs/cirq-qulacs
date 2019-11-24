import re
import collections
from typing import cast, Dict, Iterator, List, Union

import numpy as np
import qulacs
from cirq import circuits, ops, protocols, linalg
from cirq.sim import density_matrix_utils
from cirq import DensityMatrixSimulator
from cirq import DensityMatrixStepResult

class QulacsDensityMatrixSimulator(DensityMatrixSimulator):
    def _base_iterator(
                self,
                circuit: circuits.Circuit,
                qubit_order: ops.QubitOrderOrList,
                initial_state: Union[int, np.ndarray],
                perform_measurements: bool = True) -> Iterator:

        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            circuit.all_qubits())
        num_qubits = len(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        matrix = density_matrix_utils.to_valid_density_matrix(
            initial_state, num_qubits, dtype = self._dtype)
        if len(circuit) == 0:
            yield DensityMatrixStepResult(matrix, {}, qubit_map, dtype = self._dtype)

        matrix = np.reshape(matrix, (2**num_qubits, 2**num_qubits))
        noisy_moments = self.noise.noisy_moments(circuit,
                                                sorted(circuit.all_qubits()))

        state = qulacs.DensityMatrix(num_qubits)
        state.load(matrix)

        for moment in noisy_moments:
            measurements = collections.defaultdict(
                list)  # type: Dict[str, List[bool]]
            operations = moment.operations
            for op in operations:
                indices = [num_qubits - 1 - qubit_map[qubit] for qubit in op.qubits]
                indices.reverse()

                if isinstance(op, ops.MeasurementGate):
                    # Not implemented
                    raise NotImplementedError("Measurement is not supported in qulacs simulator")

                else:
                    gate = cast(ops.GateOperation, op).gate
                    channel = protocols.channel(gate)

                    qulacs_gates = []
                    for krauss in channel:
                        krauss = krauss.astype(np.complex128)
                        qulacs_gate = qulacs.gate.DenseMatrix(indices, krauss)
                        qulacs_gates.append(qulacs_gate)
                    qulacs_cptp_map = qulacs.gate.CPTP(qulacs_gates)
                    qulacs_cptp_map.update_quantum_state(state)

            matrix = state.get_matrix()
            matrix = np.reshape(matrix, (2,) * num_qubits * 2)

            yield DensityMatrixStepResult(
                    density_matrix=matrix,
                    measurements=measurements,
                    qubit_map=qubit_map,
                    dtype=self._dtype)

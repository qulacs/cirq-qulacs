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

        def on_stuck(bad_op: ops.Operation):
            return TypeError(
                "Can't simulate operations that don't implement "
                "SupportsUnitary, SupportsConsistentApplyUnitary, "
                "SupportsMixture, SupportsChannel or is a measurement: {!r}".
                format(bad_op))

        def keep(potential_op: ops.Operation) -> bool:
            return (protocols.has_channel(potential_op) or
                    isinstance(potential_op.gate, ops.MeasurementGate))


        matrix = np.reshape(matrix, (2**num_qubits, 2**num_qubits))
        noisy_moments = self.noise.noisy_moments(circuit,
                                                sorted(circuit.all_qubits()))

        state = qulacs.DensityMatrix(num_qubits)
        state.load(matrix)

        for moment in noisy_moments:
            measurements = collections.defaultdict(
                list)  # type: Dict[str, List[int]]

            channel_ops_and_measurements = protocols.decompose(
                moment, keep=keep, on_stuck_raise=on_stuck)

            for op in channel_ops_and_measurements:
                #indices = [qubit_map[qubit] for qubit in op.qubits]
                indices = [num_qubits - 1 - qubit_map[qubit] for qubit in op.qubits]
                indices.reverse()

                meas = isinstance(op.gate, ops.MeasurementGate)
                if meas:
                    # Not implemented
                    raise NotImplementedError("Measurement is not supported in qulacs simulator")

                else:
                    # TODO: Use apply_channel similar to apply_unitary.
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

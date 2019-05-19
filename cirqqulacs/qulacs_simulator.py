import re
import collections
from typing import Dict, Iterator, List, Union

import numpy as np
import qulacs
from cirq import circuits, ops, protocols
from cirq.sim import wave_function
from cirq import Simulator, SparseSimulatorStep


# Mutable named tuple to hold state and a buffer.
class _StateAndBuffer():

    def __init__(self, state, buffer):
        self.state = state
        self.buffer = buffer


class QulacsSimulator(Simulator):

    def _base_iterator(
            self,
            circuit: circuits.Circuit,
            qubit_order: ops.QubitOrderOrList,
            initial_state: Union[int, np.ndarray],
            perform_measurements: bool = True,
    ) -> Iterator:
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            circuit.all_qubits())
        num_qubits = len(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        state = wave_function.to_valid_state_vector(initial_state,
                                                    num_qubits,
                                                    self._dtype)

        if len(circuit) == 0:
            yield SparseSimulatorStep(state, {}, qubit_map, self._dtype)

        def on_stuck(bad_op: ops.Operation):
            return TypeError(
                "Can't simulate unknown operations that don't specify a "
                "_unitary_ method, a _decompose_ method, "
                "(_has_unitary_ + _apply_unitary_) methods,"
                "(_has_mixture_ + _mixture_) methods, or are measurements."

                ": {!r}".format(bad_op))

        def keep(potential_op: ops.Operation) -> bool:
            # The order of this is optimized to call has_xxx methods first.
            return (protocols.has_unitary(potential_op)
                    or protocols.has_mixture(potential_op)
                    or protocols.is_measurement(potential_op))

        data = _StateAndBuffer(
            state=np.reshape(state, (2,) * num_qubits),
            buffer=np.empty((2,) * num_qubits, dtype=self._dtype))

        shape = np.array(data.state).shape

        # Qulacs
        qulacs_flag = 0
        qulacs_state = qulacs.QuantumState(int(num_qubits))
        qulacs_circuit = qulacs.QuantumCircuit(int(num_qubits))

        for moment in circuit:

            measurements = collections.defaultdict(
                list)  # type: Dict[str, List[bool]]

            non_display_ops = (op for op in moment
                               if not isinstance(op, (ops.SamplesDisplay,
                                                      ops.WaveFunctionDisplay,
                                                      ops.DensityMatrixDisplay
                                                      )))

            unitary_ops_and_measurements = protocols.decompose(
                non_display_ops,
                keep=keep,
                on_stuck_raise=on_stuck)

            for op in unitary_ops_and_measurements:
                indices = [qubit_map[qubit] for qubit in op.qubits]
                if protocols.has_unitary(op):

                    gate_indexes = re.findall(r'([0-9]+)', str(op.qubits))

                    # single qubit unitary gates
                    if isinstance(op.gate, ops.pauli_gates._PauliX):
                        qulacs_circuit.add_X_gate(num_qubits - 1 - int(gate_indexes[0]))
                    elif isinstance(op.gate, ops.pauli_gates._PauliY):
                        qulacs_circuit.add_Y_gate(num_qubits - 1 - int(gate_indexes[0]))
                    elif isinstance(op.gate, ops.pauli_gates._PauliZ):
                        qulacs_circuit.add_Z_gate(num_qubits - 1 - int(gate_indexes[0]))
                    elif isinstance(op.gate, ops.common_gates.HPowGate):
                        qulacs_circuit.add_H_gate(num_qubits - 1 - int(gate_indexes[0]))
                    elif isinstance(op.gate, ops.common_gates.XPowGate):
                        qulacs_circuit.add_dense_matrix_gate(num_qubits - 1 - int(gate_indexes[0]),
                                                             op._unitary_())
                    elif isinstance(op.gate, ops.common_gates.YPowGate):
                        qulacs_circuit.add_dense_matrix_gate(num_qubits - 1 - int(gate_indexes[0]),
                                                             op._unitary_())
                    elif isinstance(op.gate, ops.common_gates.ZPowGate):
                        qulacs_circuit.add_dense_matrix_gate(num_qubits - 1 - int(gate_indexes[0]),
                                                             op._unitary_())
                    elif isinstance(op.gate, circuits.qasm_output.QasmUGate):
                        qulacs_circuit.add_dense_matrix_gate(num_qubits - 1 - int(gate_indexes[0]),
                                                             op._unitary_())

                    # Two Qubit Unitary Gates
                    elif isinstance(op.gate, ops.common_gates.CNotPowGate):
                        qulacs_circuit.add_CNOT_gate(num_qubits - 1 - int(gate_indexes[0]),
                                                     num_qubits - 1 - int(gate_indexes[1]))
                    elif isinstance(op.gate, ops.common_gates.CZPowGate):
                        qulacs_circuit.add_CZ_gate(num_qubits - 1 - int(gate_indexes[0]),
                                                   num_qubits - 1 - int(gate_indexes[1]))
                    elif isinstance(op.gate, ops.common_gates.SwapPowGate):
                        qulacs_circuit.add_SWAP_gate(num_qubits - 1 - int(gate_indexes[0]),
                                                     num_qubits - 1 - int(gate_indexes[1]))
                    elif isinstance(op.gate, ops.parity_gates.XXPowGate):
                        qulacs_circuit.add_dense_matrix_gate(
                            [num_qubits - 1 - int(gate_indexes[0]),
                             num_qubits - 1 - int(gate_indexes[1])],
                            op._unitary_())
                    elif isinstance(op.gate, ops.parity_gates.YYPowGate):
                        qulacs_circuit.add_dense_matrix_gate(
                            [num_qubits - 1 - int(gate_indexes[0]),
                             num_qubits - 1 - int(gate_indexes[1])],
                            op._unitary_())
                    elif isinstance(op.gate, ops.parity_gates.ZZPowGate):
                        qulacs_circuit.add_dense_matrix_gate(
                            [num_qubits - 1 - int(gate_indexes[0]),
                             num_qubits - 1 - int(gate_indexes[1])],
                            op._unitary_())

                    # Three Qubit Unitary Gates
                    elif isinstance(op.gate, ops.three_qubit_gates.CCXPowGate):
                        qulacs_circuit.add_dense_matrix_gate(
                            [num_qubits - 1 - int(gate_indexes[0]),
                             num_qubits - 1 - int(gate_indexes[1]),
                             num_qubits - 1 - int(gate_indexes[2])],
                            op._unitary_())
                    elif isinstance(op.gate, ops.three_qubit_gates.CCZPowGate):
                        qulacs_circuit.add_dense_matrix_gate(
                            [num_qubits - 2 - int(gate_indexes[0]),
                             num_qubits - 1 - int(gate_indexes[1]),
                             num_qubits - 1 - int(gate_indexes[2])],
                            op._unitary_())

                    qulacs_flag = 1

                elif protocols.is_measurement(op):
                    # Do measurements second, since there may be mixtures that
                    # operate as measurements.
                    # TODO: support measurement outside the computational basis.

                    if perform_measurements:
                        if qulacs_flag == 1:
                            self._simulate_on_qulacs(data, shape, qulacs_state, qulacs_circuit)
                            qulacs_flag = 0
                        self._simulate_measurement(op, data, indices,
                                                   measurements, num_qubits)

                elif protocols.has_mixture(op):
                    self._simulate_mixture(op, data, indices)

        if qulacs_flag == 1:
            self._simulate_on_qulacs(data, shape, qulacs_state, qulacs_circuit)
            qulacs_flag = 0

        del qulacs_state
        del qulacs_circuit

        yield SparseSimulatorStep(
            state_vector=data.state,
            measurements=measurements,
            qubit_map=qubit_map,
            dtype=self._dtype)


    def _simulate_on_qulacs(
            self,
            data: _StateAndBuffer,
            shape: tuple,
            qulacs_state: qulacs.QuantumState,
            qulacs_circuit: qulacs.QuantumCircuit,
    ) -> None:
        data.buffer = data.state
        cirq_state = np.array(data.state).flatten().astype(np.complex64)
        qulacs_state.load(cirq_state)
        qulacs_circuit.update_quantum_state(qulacs_state)
        data.state = qulacs_state.get_vector().reshape(shape)


class QulacsSimulatorGpu(Simulator):

    def _base_iterator(
            self,
            circuit: circuits.Circuit,
            qubit_order: ops.QubitOrderOrList,
            initial_state: Union[int, np.ndarray],
            perform_measurements: bool = True,
    ) -> Iterator:
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            circuit.all_qubits())
        num_qubits = len(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        state = wave_function.to_valid_state_vector(initial_state,
                                                    num_qubits,
                                                    self._dtype)

        if len(circuit) == 0:
            yield SparseSimulatorStep(state, {}, qubit_map, self._dtype)

        def on_stuck(bad_op: ops.Operation):
            return TypeError(
                "Can't simulate unknown operations that don't specify a "
                "_unitary_ method, a _decompose_ method, "
                "(_has_unitary_ + _apply_unitary_) methods,"
                "(_has_mixture_ + _mixture_) methods, or are measurements."

                ": {!r}".format(bad_op))

        def keep(potential_op: ops.Operation) -> bool:
            # The order of this is optimized to call has_xxx methods first.
            return (protocols.has_unitary(potential_op)
                    or protocols.has_mixture(potential_op)
                    or protocols.is_measurement(potential_op))

        data = _StateAndBuffer(
            state=np.reshape(state, (2,) * num_qubits),
            buffer=np.empty((2,) * num_qubits, dtype=self._dtype))

        shape = np.array(data.state).shape

        # Qulacs
        qulacs_flag = 0
        qulacs_state = qulacs.QuantumStateGpu(int(num_qubits))
        qulacs_circuit = qulacs.QuantumCircuit(int(num_qubits))

        for moment in circuit:

            measurements = collections.defaultdict(
                list)  # type: Dict[str, List[bool]]

            non_display_ops = (op for op in moment
                               if not isinstance(op, (ops.SamplesDisplay,
                                                      ops.WaveFunctionDisplay,
                                                      ops.DensityMatrixDisplay
                                                      )))

            unitary_ops_and_measurements = protocols.decompose(
                non_display_ops,
                keep=keep,
                on_stuck_raise=on_stuck)

            for op in unitary_ops_and_measurements:
                indices = [qubit_map[qubit] for qubit in op.qubits]
                if protocols.has_unitary(op):

                    gate_indexes = re.findall(r'([0-9]+)', str(op.qubits))

                    # single qubit unitary gates
                    if isinstance(op.gate, ops.pauli_gates._PauliX):
                        qulacs_circuit.add_X_gate(num_qubits - 1 - int(gate_indexes[0]))
                    elif isinstance(op.gate, ops.pauli_gates._PauliY):
                        qulacs_circuit.add_Y_gate(num_qubits - 1 - int(gate_indexes[0]))
                    elif isinstance(op.gate, ops.pauli_gates._PauliZ):
                        qulacs_circuit.add_Z_gate(num_qubits - 1 - int(gate_indexes[0]))
                    elif isinstance(op.gate, ops.common_gates.HPowGate):
                        qulacs_circuit.add_H_gate(num_qubits - 1 - int(gate_indexes[0]))
                    elif isinstance(op.gate, ops.common_gates.XPowGate):
                        qulacs_circuit.add_dense_matrix_gate(num_qubits - 1 - int(gate_indexes[0]),
                                                             op._unitary_())
                    elif isinstance(op.gate, ops.common_gates.YPowGate):
                        qulacs_circuit.add_dense_matrix_gate(num_qubits - 1 - int(gate_indexes[0]),
                                                             op._unitary_())
                    elif isinstance(op.gate, ops.common_gates.ZPowGate):
                        qulacs_circuit.add_dense_matrix_gate(num_qubits - 1 - int(gate_indexes[0]),
                                                             op._unitary_())
                    elif isinstance(op.gate, circuits.qasm_output.QasmUGate):
                        qulacs_circuit.add_dense_matrix_gate(num_qubits - 1 - int(gate_indexes[0]),
                                                             op._unitary_())

                    # Two Qubit Unitary Gates
                    elif isinstance(op.gate, ops.common_gates.CNotPowGate):
                        qulacs_circuit.add_CNOT_gate(num_qubits - 1 - int(gate_indexes[0]),
                                                     num_qubits - 1 - int(gate_indexes[1]))
                    elif isinstance(op.gate, ops.common_gates.CZPowGate):
                        qulacs_circuit.add_CZ_gate(num_qubits - 1 - int(gate_indexes[0]),
                                                   num_qubits - 1 - int(gate_indexes[1]))
                    elif isinstance(op.gate, ops.common_gates.SwapPowGate):
                        qulacs_circuit.add_SWAP_gate(num_qubits - 1 - int(gate_indexes[0]),
                                                     num_qubits - 1 - int(gate_indexes[1]))
                    elif isinstance(op.gate, ops.parity_gates.XXPowGate):
                        qulacs_circuit.add_dense_matrix_gate(
                            [num_qubits - 1 - int(gate_indexes[0]),
                             num_qubits - 1 - int(gate_indexes[1])],
                            op._unitary_())
                    elif isinstance(op.gate, ops.parity_gates.YYPowGate):
                        qulacs_circuit.add_dense_matrix_gate(
                            [num_qubits - 1 - int(gate_indexes[0]),
                             num_qubits - 1 - int(gate_indexes[1])],
                            op._unitary_())
                    elif isinstance(op.gate, ops.parity_gates.ZZPowGate):
                        qulacs_circuit.add_dense_matrix_gate(
                            [num_qubits - 1 - int(gate_indexes[0]),
                             num_qubits - 1 - int(gate_indexes[1])],
                            op._unitary_())

                    # Three Qubit Unitary Gates
                    elif isinstance(op.gate, ops.three_qubit_gates.CCXPowGate):
                        qulacs_circuit.add_dense_matrix_gate(
                            [num_qubits - 1 - int(gate_indexes[0]),
                             num_qubits - 1 - int(gate_indexes[1]),
                             num_qubits - 1 - int(gate_indexes[2])],
                            op._unitary_())
                    elif isinstance(op.gate, ops.three_qubit_gates.CCZPowGate):
                        qulacs_circuit.add_dense_matrix_gate(
                            [num_qubits - 2 - int(gate_indexes[0]),
                             num_qubits - 1 - int(gate_indexes[1]),
                             num_qubits - 1 - int(gate_indexes[2])],
                            op._unitary_())

                    qulacs_flag = 1

                elif protocols.is_measurement(op):
                    # Do measurements second, since there may be mixtures that
                    # operate as measurements.
                    # TODO: support measurement outside the computational basis.

                    if perform_measurements:
                        if qulacs_flag == 1:
                            self._simulate_on_qulacs(data, shape, qulacs_state, qulacs_circuit)
                            qulacs_flag = 0
                        self._simulate_measurement(op, data, indices,
                                                   measurements, num_qubits)

                elif protocols.has_mixture(op):
                    self._simulate_mixture(op, data, indices)

        if qulacs_flag == 1:
            self._simulate_on_qulacs(data, shape, qulacs_state, qulacs_circuit)
            qulacs_flag = 0

        del qulacs_state
        del qulacs_circuit

        yield SparseSimulatorStep(
            state_vector=data.state,
            measurements=measurements,
            qubit_map=qubit_map,
            dtype=self._dtype)

    def _simulate_on_qulacs(
            self,
            data: _StateAndBuffer,
            shape: tuple,
            qulacs_state: qulacs.QuantumState,
            qulacs_circuit: qulacs.QuantumCircuit,
    ) -> None:
        data.buffer = data.state
        cirq_state = np.array(data.state).flatten().astype(np.complex64)
        qulacs_state.load(cirq_state)
        qulacs_circuit.update_quantum_state(qulacs_state)
        data.state = qulacs_state.get_vector().reshape(shape)

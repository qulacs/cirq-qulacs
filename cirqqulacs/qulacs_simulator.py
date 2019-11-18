from typing import Dict, Iterator, List, Union, Any, Type, cast

import numpy as np
import collections
import qulacs
from cirq import circuits, ops, protocols, schedules, study, value
from cirq.sim import SimulatesFinalState, SimulationTrialResult, wave_function

def _get_google_rotx(exponent : float) -> np.ndarray:
    rot = exponent
    g = np.exp(1.j*np.pi*rot/2)
    c = np.cos(np.pi*rot/2)
    s = np.sin(np.pi*rot/2)
    mat = np.array([
        [g*c, -1.j*g*s],
        [-1.j*g*s, g*c]
    ])
    return mat

def _get_google_rotz(exponent : float) -> np.ndarray:
    return np.diag([1., np.exp(1.j*np.pi*exponent)])

class QulacsSimulator(SimulatesFinalState):
    def __init__(self, *, 
            dtype: Type[np.number] = np.complex128, 
            seed: value.RANDOM_STATE_LIKE = None):
        self._dtype = dtype
        self._prng = value.parse_random_state(seed)

    def _get_qulacs_state(self, num_qubits: int):
        return qulacs.QuantumState(num_qubits)

    def simulate_sweep(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        params: study.Sweepable,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> List['SimulationTrialResult']:
        """Simulates the supplied Circuit or Schedule with Qulacs
        Args:
            program: The circuit or schedule to simulate.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation.  See
                documentation of the implementing class for details.
        Returns:
            List of SimulationTrialResults for this run, one for each
            possible parameter resolver.
        """        
        trial_results = []
        # sweep for each parameters
        resolvers = study.to_resolvers(params)
        for resolver in resolvers:

            # result circuit
            cirq_circuit = protocols.resolve_parameters(program, resolver)
            qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(cirq_circuit.all_qubits())
            qubit_map = {q: i for i, q in enumerate(qubits)}
            num_qubits = len(qubits)

            # create state
            qulacs_state = self._get_qulacs_state(num_qubits)
            if initial_state is not None:
                cirq_state = wave_function.to_valid_state_vector(initial_state,num_qubits)
                qulacs_state.load(cirq_state)
                del cirq_state

            # create circuit

            ## function called when circuit contains not tractable operations
            def on_stuck(bad_op: ops.Operation):
                return TypeError(
                    "Can't simulate unknown operations that don't specify a "
                    "_unitary_ method, a _decompose_ method, "
                    "(_has_unitary_ + _apply_unitary_) methods,"
                    "(_has_mixture_ + _mixture_) methods, or are measurements."
                    ": {!r}".format(bad_op))

            ## return True if operation is tractable with qulacs
            def keep(potential_op: ops.Operation) -> bool:
                return (protocols.has_unitary(potential_op) or
                        protocols.has_mixture(potential_op) or
                        protocols.is_measurement(potential_op) or
                        isinstance(potential_op.gate, ops.ResetChannel))

            qulacs_circuit = qulacs.QuantumCircuit(num_qubits)
            address_to_key = {}
            register_address = 0
            for moment in cirq_circuit:
                operations = protocols.decompose(
                    moment, keep=keep, on_stuck_raise=on_stuck)

                for op in operations:
                    # In qulacs, the lowest bit is right, though it is left in cirq.
                    # we need to invert indices
                    indices = [num_qubits - 1 - qubit_map[qubit] for qubit in op.qubits]

                    if isinstance(op.gate, ops.ResetChannel):
                        qulacs_circuit.update_quantum_state(qulacs_state)
                        qulacs_state.set_zero_state()
                        qulacs_circuit = qulacs.QuantumCircuit(num_qubits)

                    elif protocols.has_unitary(op):
                        self._append_gate(op, qulacs_circuit, indices)

                    elif protocols.is_measurement(op):
                        for index in indices:
                            qulacs_circuit.add_gate(qulacs.gate.Measurement(index, register_address))
                            address_to_key[register_address] = protocols.measurement_key(op.gate)
                            register_address += 1

                    elif protocols.has_mixture(op):
                        indices.reverse()
                        qulacs_gates = []
                        gate = cast(ops.GateOperation, op).gate
                        channel = protocols.channel(gate)
                        for krauss in channel:
                            krauss = krauss.astype(np.complex128)
                            qulacs_gate = qulacs.gate.DenseMatrix(indices, krauss)
                            qulacs_gates.append(qulacs_gate)
                        qulacs_cptp_map = qulacs.gate.CPTP(qulacs_gates)
                        qulacs.circuit.add_gate(qulacs_cptp_map)

            # perform simulation
            qulacs_circuit.update_quantum_state(qulacs_state)

            # fetch final state and measurement results
            final_state = qulacs_state.get_vector()
            measurements = collections.defaultdict(list)
            for register_index in range(register_address):
                key = address_to_key[register_index]
                value = qulacs_state.get_classical_value(register_index)
                measurements[key].append(value)

            # create result for this parameter
            result = SimulationTrialResult(
                params = resolver,
                measurements = measurements,
                final_simulator_state = final_state
            )
            trial_results.append(result)

            # release memory
            del qulacs_state
            del qulacs_circuit

        return trial_results

    def _append_gate(self, op : ops.GateOperation, qulacs_circuit : qulacs.QuantumCircuit, indices : np.array):
        # One qubit gate
        if isinstance(op.gate, ops.pauli_gates._PauliX):
            qulacs_circuit.add_X_gate(indices[0])
        elif isinstance(op.gate, ops.pauli_gates._PauliY):
            qulacs_circuit.add_Y_gate(indices[0])
        elif isinstance(op.gate, ops.pauli_gates._PauliZ):
            qulacs_circuit.add_Z_gate(indices[0])
        elif isinstance(op.gate, ops.common_gates.HPowGate):
            qulacs_circuit.add_H_gate(indices[0])
        elif isinstance(op.gate, ops.common_gates.XPowGate):
            qulacs_circuit.add_RX_gate(indices[0], -np.pi*op.gate._exponent)
        elif isinstance(op.gate, ops.common_gates.YPowGate):
            qulacs_circuit.add_RY_gate(indices[0], -np.pi*op.gate._exponent)
        elif isinstance(op.gate, ops.common_gates.ZPowGate):
            qulacs_circuit.add_RZ_gate(indices[0], -np.pi*op.gate._exponent)

        # Two qubit gate
        elif isinstance(op.gate, ops.common_gates.CNotPowGate):
            if op.gate._exponent == 1.0:
                qulacs_circuit.add_CNOT_gate(indices[0], indices[1])
            else:
                mat = _get_google_rotx(op.gate._exponent)
                gate = qulacs.gate.DenseMatrix(indices[1], mat)
                gate.add_control_qubit(indices[0],1)
                qulacs_circuit.add_gate(gate)
        elif isinstance(op.gate, ops.common_gates.CZPowGate):
            if op.gate._exponent == 1.0:
                qulacs_circuit.add_CZ_gate(indices[0], indices[1])
            else:
                mat = _get_google_rotz(op.gate._exponent)
                gate = qulacs.gate.DenseMatrix(indices[1], mat)
                gate.add_control_qubit(indices[0],1)
                qulacs_circuit.add_gate(gate)
        elif isinstance(op.gate, ops.common_gates.SwapPowGate):
            if op.gate._exponent == 1.0:
                qulacs_circuit.add_SWAP_gate(indices[0], indices[1])
            else:
                qulacs_circuit.add_dense_matrix_gate(indices,op._unitary_())
        elif isinstance(op.gate, ops.parity_gates.XXPowGate):
            qulacs_circuit.add_multi_Pauli_rotation_gate(indices, [1,1], -np.pi*op.gate._exponent)
        elif isinstance(op.gate, ops.parity_gates.YYPowGate):
            qulacs_circuit.add_multi_Pauli_rotation_gate(indices, [2,2], -np.pi*op.gate._exponent)
        elif isinstance(op.gate, ops.parity_gates.ZZPowGate):
            qulacs_circuit.add_multi_Pauli_rotation_gate(indices, [3,3], -np.pi*op.gate._exponent)

        # Three qubit gate
        elif isinstance(op.gate, ops.three_qubit_gates.CCXPowGate):
            mat = _get_google_rotx(op.gate._exponent)
            gate = qulacs.gate.DenseMatrix(indices[2], mat)
            gate.add_control_qubit(indices[0],1)
            gate.add_control_qubit(indices[1],1)
            qulacs_circuit.add_gate(gate)
        elif isinstance(op.gate, ops.three_qubit_gates.CCZPowGate):
            mat = _get_google_rotz(op.gate._exponent)
            gate = qulacs.gate.DenseMatrix(indices[2], mat)
            gate.add_control_qubit(indices[0],1)
            gate.add_control_qubit(indices[1],1)
            qulacs_circuit.add_gate(gate)
        elif isinstance(op.gate, ops.three_qubit_gates.CSwapGate):
            mat = np.zeros(shape=(4,4))
            mat[0,0] = 1; mat[1,2] = 1; mat[2,1] = 1; mat[3,3] = 1
            gate = qulacs.gate.DenseMatrix(indices[1:], mat)
            gate.add_control_qubit(indices[0],1)
            qulacs_circuit.add_gate(gate)

        # Misc
        else:
            indices.reverse()
            qulacs_circuit.add_dense_matrix_gate(indices,op._unitary_())


class QulacsSimulatorGpu(QulacsSimulator):
    def _get_qulacs_state(self, num_qubits : int):
        try:
            state = qulacs.QuantumStateGpu(num_qubits)
            return state
        except AttributeError:
            raise Exception("GPU simulator is not installed")


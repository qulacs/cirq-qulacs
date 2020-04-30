import re
import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import cirq
from cirqqulacs import QulacsDensityMatrixSimulator
from .test_qulacs_simulator import parse_qasm_to_QulacsCircuit

class TestQulacsDensityMatrixSimulator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.qubit_n = 5
        self.test_repeat = 4

    def check_result(self, circuit, rtol=1e-9, atol=0, dtype=np.complex128):
        qulacs_result = QulacsDensityMatrixSimulator(dtype=dtype).simulate(circuit)
        actual = qulacs_result.final_density_matrix
        cirq_result = cirq.DensityMatrixSimulator(dtype=dtype).simulate(circuit)
        expected = cirq_result.final_density_matrix
        assert_allclose(actual, expected, rtol=1e-5, atol=0)

    def check_single_qubit_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        for _ in range(self.test_repeat):
            index = np.random.randint(self.qubit_n)
            circuit.append(gate_op(qubits[index]))
            self.check_result(circuit)

    def test_QulacsDensityMatrixSimulator_Xgate(self):
        self.check_single_qubit_gate(cirq.ops.X)

    def test_QulacsDensityMatrixSimulator_Ygate(self):
        self.check_single_qubit_gate(cirq.ops.Y)

    def test_QulacsDensityMatrixSimulator_Zgate(self):
        self.check_single_qubit_gate(cirq.ops.Z)

    def test_QulacsDensityMatrixSimulator_Hgate(self):
        self.check_single_qubit_gate(cirq.ops.H)

    def test_QulacsDensityMatrixSimulator_Sgate(self):
        self.check_single_qubit_gate(cirq.ops.S)

    def test_QulacsDensityMatrixSimulator_Tgate(self):
        self.check_single_qubit_gate(cirq.ops.T)

    def check_single_qubit_rotation_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        for _ in range(self.test_repeat):
            index = np.random.randint(self.qubit_n)
            angle = np.random.rand()*np.pi*2
            circuit.append(gate_op(angle).on(qubits[index]))
            self.check_result(circuit)

    def test_QulacsDensityMatrixSimulator_RXgate(self):
        self.check_single_qubit_rotation_gate(cirq.rx)

    def test_QulacsDensityMatrixSimulator_RYgate(self):
        self.check_single_qubit_rotation_gate(cirq.ry)

    def test_QulacsDensityMatrixSimulator_RZgate(self):
        self.check_single_qubit_rotation_gate(cirq.rz)

    def test_QulacsDensityMatrixSimulator_Ugate(self):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        for _ in range(self.test_repeat):
            index = np.random.randint(self.qubit_n)
            angle = np.random.rand(3)*np.pi*2
            #circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
            circuit.append(cirq.rz(angle[0]).on(qubits[index]))
            circuit.append(cirq.ry(angle[1]).on(qubits[index]))
            circuit.append(cirq.rz(angle[2]).on(qubits[index]))
            self.check_result(circuit)
       
    def check_two_qubit_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        all_indices = np.arange(self.qubit_n)
        for _ in range(self.test_repeat):
            for index in range(self.qubit_n):
                angle = np.random.rand(3)*np.pi*2
                #circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
                circuit.append(cirq.rz(angle[0]).on(qubits[index]))
                circuit.append(cirq.ry(angle[1]).on(qubits[index]))
                circuit.append(cirq.rz(angle[2]).on(qubits[index]))
            np.random.shuffle(all_indices)
            index = all_indices[:2]
            circuit.append(gate_op(qubits[index[0]],qubits[index[1]]))
            self.check_result(circuit)

    def test_QulacsDensityMatrixSimulator_CNOTgate(self):
        self.check_two_qubit_gate(cirq.ops.CNOT)

    def test_QulacsDensityMatrixSimulator_CZgate(self):
        self.check_two_qubit_gate(cirq.ops.CZ)

    def test_QulacsDensityMatrixSimulator_SWAPgate(self):
        self.check_two_qubit_gate(cirq.ops.SWAP)

    def test_QulacsDensityMatrixSimulator_XXgate(self):
        self.check_two_qubit_gate(cirq.ops.XX)

    def test_QulacsDensityMatrixSimulator_YYgate(self):
        self.check_two_qubit_gate(cirq.ops.YY)

    def test_QulacsDensityMatrixSimulator_ZZgate(self):
        self.check_two_qubit_gate(cirq.ops.ZZ)

    def check_three_qubit_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        all_indices = np.arange(self.qubit_n)
        for _ in range(self.test_repeat):
            for index in range(self.qubit_n):
                angle = np.random.rand(3)*np.pi*2
                #circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
                circuit.append(cirq.rz(angle[0]).on(qubits[index]))
                circuit.append(cirq.ry(angle[1]).on(qubits[index]))
                circuit.append(cirq.rz(angle[2]).on(qubits[index]))
            np.random.shuffle(all_indices)
            index = all_indices[:3]
            circuit.append(gate_op(qubits[index[0]],qubits[index[1]],qubits[index[2]]))
            self.check_result(circuit)

    def test_QulacsDensityMatrixSimulator_CCXgate(self):
        self.check_three_qubit_gate(cirq.ops.CCX)

    def test_QulacsDensityMatrixSimulator_CCZgate(self):
        self.check_three_qubit_gate(cirq.ops.CCZ)

    def test_QulacsDensityMatrixSimulator_TOFFOLIgate(self):
        self.check_three_qubit_gate(cirq.ops.TOFFOLI)

    def test_QulacsDensityMatrixSimulator_QuantumVolume(self):
        qubit_n = 6
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        parse_qasm_to_QulacsCircuit('tests/quantum_volume_n6_d8_0_9.qasm', circuit, qubits)
        self.check_result(circuit)

if __name__ == "__main__":
    unittest.main()

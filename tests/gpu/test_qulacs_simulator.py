import re
import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.stats import unitary_group

import cirq
from cirqqulacs import QulacsSimulatorGpu



def parse_qasm_to_QulacsCircuit(input_filename,cirq_circuit,cirq_qubits):

    with open(input_filename, "r") as ifile:
        lines = ifile.readlines()
 
        for line in lines:
            s = re.search(r"qreg|cx|u3|u1", line)
 
            if s is None:
                continue
 
            elif s.group() == 'qreg':
                match = re.search(r"\d\d*", line)
                # print(match)
                continue
 
            elif s.group() == 'cx':
                match = re.findall(r'\[\d\d*\]', line)
                c_qbit = int(match[0].strip('[]'))
                t_qbit = int(match[1].strip('[]'))
                cirq_circuit.append(cirq.ops.CNOT(cirq_qubits[c_qbit],cirq_qubits[t_qbit]))
                continue
 
            elif s.group() == 'u3':
                m_r = re.findall(r'[-]?\d\.\d\d*', line)
                m_i = re.findall(r'\[\d\d*\]', line)
 
                cirq_circuit.append(cirq.circuits.qasm_output.QasmUGate(float(m_r[0]),float(m_r[1]),float(m_r[2])).on(cirq_qubits[int(m_i[0].strip('[]'))]))
 
                continue
 
            elif s.group() == 'u1':
                m_r = re.findall(r'[-]?\d\.\d\d*', line)
                m_i = re.findall(r'\[\d\d*\]', line)
 
                cirq_circuit.append(cirq.circuits.qasm_output.QasmUGate(float(m_r[0]), 0, 0).on(cirq_qubits[int(m_i[0].strip('[]'))]))
 
                continue



class TestQulacsSimulator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.qubit_n = 5
        self.test_repeat = 4

    def compare_state_vector(self, actual, expected):
        actual_2d = np.atleast_2d(actual)
        actual_mat = actual_2d.T.conj() @ actual_2d

        expected_2d = np.atleast_2d(expected)
        expected_mat = expected_2d.T.conj() @ expected_2d
        assert_allclose(actual_mat, expected_mat, rtol=1e-5, atol=0)

    def check_result(self, circuit, rtol=1e-9, atol=0, dtype=np.complex128):
        qulacs_result = QulacsSimulatorGpu(dtype=dtype).simulate(circuit)
        actual = qulacs_result._final_simulator_state
        cirq_result = cirq.Simulator(dtype=dtype).simulate(circuit)
        expected = cirq_result.final_state
        self.compare_state_vector(actual, expected)


    def check_single_qubit_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        for _ in range(self.test_repeat):
            index = np.random.randint(self.qubit_n)
            circuit.append(gate_op(qubits[index]))
            #print("flip {}".format(index))
            self.check_result(circuit)


    def check_single_qubit_rotation_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        for _ in range(self.test_repeat):
            index = np.random.randint(self.qubit_n)
            angle = np.random.rand()*np.pi*2
            circuit.append(gate_op(angle).on(qubits[index]))
            self.check_result(circuit)


    def check_two_qubit_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        all_indices = np.arange(self.qubit_n)
        for _ in range(self.test_repeat):
            for index in range(self.qubit_n):
                angle = np.random.rand(3)*np.pi*2
                circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
            np.random.shuffle(all_indices)
            index = all_indices[:2]
            circuit.append(gate_op(qubits[index[0]],qubits[index[1]]))
            self.check_result(circuit)


    def check_two_qubit_rotation_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        all_indices = np.arange(self.qubit_n)
        for _ in range(self.test_repeat):
            for index in range(self.qubit_n):
                angle = np.random.rand(3)*np.pi*2
                circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
            np.random.shuffle(all_indices)
            index = all_indices[:2]
            angle = np.random.rand()*np.pi*2
            gate_op_angle = gate_op(exponent=angle)
            circuit.append(gate_op_angle(qubits[index[0]], qubits[index[1]]))
            self.check_result(circuit)

    def check_three_qubit_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        all_indices = np.arange(self.qubit_n)
        for _ in range(self.test_repeat):
            for index in range(self.qubit_n):
                angle = np.random.rand(3)*np.pi*2
                circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
            np.random.shuffle(all_indices)
            index = all_indices[:3]
            circuit.append(gate_op(qubits[index[0]],qubits[index[1]],qubits[index[2]]))
            self.check_result(circuit)



    def check_three_qubit_rotation_gate(self, gate_op):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        all_indices = np.arange(self.qubit_n)
        for _ in range(self.test_repeat):
            for index in range(self.qubit_n):
                angle = np.random.rand(3)*np.pi*2
                circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
            np.random.shuffle(all_indices)
            index = all_indices[:3]
            angle = np.random.rand()*np.pi*2
            gate_op_angle = gate_op(exponent=angle)
            circuit.append(gate_op_angle(qubits[index[0]], qubits[index[1]], qubits[index[2]]))
            self.check_result(circuit)


    def test_QulacsSimulator_Xgate(self):
        self.check_single_qubit_gate(cirq.ops.X)

    def test_QulacsSimulator_Ygate(self):
        self.check_single_qubit_gate(cirq.ops.Y)

    def test_QulacsSimulator_Zgate(self):
        self.check_single_qubit_gate(cirq.ops.Z)

    def test_QulacsSimulator_Hgate(self):
        self.check_single_qubit_gate(cirq.ops.H)

    def test_QulacsSimulator_Sgate(self):
        self.check_single_qubit_gate(cirq.ops.S)

    def test_QulacsSimulator_Tgate(self):
        self.check_single_qubit_gate(cirq.ops.T)

    def test_QulacsSimulator_RXgate(self):
        self.check_single_qubit_rotation_gate(cirq.rx)

    def test_QulacsSimulator_RYgate(self):
        self.check_single_qubit_rotation_gate(cirq.ry)

    def test_QulacsSimulator_RZgate(self):
        self.check_single_qubit_rotation_gate(cirq.rz)

    def test_QulacsSimulator_CNOTgate(self):
        self.check_two_qubit_gate(cirq.ops.CNOT)

    def test_QulacsSimulator_CZgate(self):
        self.check_two_qubit_gate(cirq.ops.CZ)

    def test_QulacsSimulator_SWAPgate(self):
        self.check_two_qubit_gate(cirq.ops.SWAP)

    def test_QulacsSimulator_ISWAPgate(self):
        self.check_two_qubit_gate(cirq.ops.ISWAP)

    def test_QulacsSimulator_XXgate(self):
        self.check_two_qubit_gate(cirq.ops.XX)

    def test_QulacsSimulator_YYgate(self):
        self.check_two_qubit_gate(cirq.ops.YY)

    def test_QulacsSimulator_ZZgate(self):
        self.check_two_qubit_gate(cirq.ops.ZZ)

    def test_QulacsSimulator_CNotPowgate(self):
        self.check_two_qubit_rotation_gate(cirq.ops.CNotPowGate)

    def test_QulacsSimulator_CZPowgate(self):
        self.check_two_qubit_rotation_gate(cirq.ops.CZPowGate)

    def test_QulacsSimulator_SwapPowgate(self):
        self.check_two_qubit_rotation_gate(cirq.ops.SwapPowGate)

    def test_QulacsSimulator_ISwapPowgate(self):
        self.check_two_qubit_rotation_gate(cirq.ops.ISwapPowGate)

    def test_QulacsSimulator_XXPowgate(self):
        self.check_two_qubit_rotation_gate(cirq.ops.XXPowGate)

    def test_QulacsSimulator_YYPowgate(self):
        self.check_two_qubit_rotation_gate(cirq.ops.YYPowGate)

    def test_QulacsSimulator_ZZPowgate(self):
        self.check_two_qubit_rotation_gate(cirq.ops.ZZPowGate)

    def test_QulacsSimulator_CCXgate(self):
        self.check_three_qubit_gate(cirq.ops.CCX)

    def test_QulacsSimulator_CCZgate(self):
        self.check_three_qubit_gate(cirq.ops.CCZ)

    def test_QulacsSimulator_CSwapGate(self):
        self.check_three_qubit_gate(cirq.ops.CSWAP)

    def test_QulacsSimulator_TOFFOLIgate(self):
        self.check_three_qubit_gate(cirq.ops.TOFFOLI)

    def test_QulacsSimulator_CCXPowgate(self):
        self.check_three_qubit_rotation_gate(cirq.ops.CCXPowGate)

    def test_QulacsSimulator_CCZPowgate(self):
        self.check_three_qubit_rotation_gate(cirq.ops.CCZPowGate)

    def test_QulacsSimulator_Ugate(self):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        for _ in range(self.test_repeat):
            index = np.random.randint(self.qubit_n)
            angle = np.random.rand(3)*np.pi*2
            circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
            self.check_result(circuit)

    def test_QulacsSimulator_SingleQubitMatrixGate(self):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        for _ in range(self.test_repeat):
            for index in range(self.qubit_n):
                angle = np.random.rand(3)*np.pi*2
                circuit.append(cirq.circuits.qasm_output.QasmUGate(angle[0], angle[1], angle[2]).on(qubits[index]))
            index = np.random.randint(self.qubit_n)
            mat = unitary_group.rvs(2)
            circuit.append(cirq.MatrixGate(mat).on(qubits[index]))
            self.check_result(circuit)

    def test_QulacsSimulator_TwoQubitMatrixGate(self):
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_n)]
        circuit = cirq.Circuit()
        all_indices = np.arange(self.qubit_n)
        for _ in range(self.test_repeat):
            np.random.shuffle(all_indices)
            index = all_indices[:2]
            mat = unitary_group.rvs(4)
            circuit.append(cirq.MatrixGate(mat).on(qubits[index[0]], qubits[index[1]]))
            self.check_result(circuit)

    def test_QulacsSimulator_QuantumVolume(self):
        qubit_n = 20
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        parse_qasm_to_QulacsCircuit('tests/quantum_volume_n10_d8_0_0.qasm', circuit, qubits)
        self.check_result(circuit)

if __name__ == "__main__":
    unittest.main()

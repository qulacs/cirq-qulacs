import re
import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import cirq
from cirqqulacs import QulacsDensityMatrixSimulator



def parse_qasm_to_QulacsCircuit(input_filename,cirq_circuit,cirq_qubits, qubit_n):

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
                if c_qbit >= qubit_n or t_qbit >= qubit_n:
                    continue
                cirq_circuit.append(cirq.ops.CNOT(cirq_qubits[c_qbit],cirq_qubits[t_qbit]))
                continue
 
            elif s.group() == 'u3':
                m_r = re.findall(r'[-]?\d\.\d\d*', line)
                m_i = re.findall(r'\[\d\d*\]', line)

                target_index = int(m_i[0].strip('[]'))
                if target_index >= qubit_n:
                    continue 
                cirq_circuit.append(cirq.circuits.qasm_output.QasmUGate(float(m_r[0]),float(m_r[1]),float(m_r[2])).on(cirq_qubits[target_index]))
 
                continue
 
            elif s.group() == 'u1':
                m_r = re.findall(r'[-]?\d\.\d\d*', line)
                m_i = re.findall(r'\[\d\d*\]', line)
                target_index = int(m_i[0].strip('[]'))
                if target_index >= qubit_n:
                    continue
                cirq_circuit.append(cirq.circuits.qasm_output.QasmUGate(float(m_r[0]), 0, 0).on(cirq_qubits[target_index]))
 
                continue



class TestQulacsDensityMatrixSimulator(unittest.TestCase):

    def test_QulacsDensityMatrixSimulator_Xgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.X(qubits[0]))
        circuit.append(cirq.ops.X(qubits[1]))
        circuit.append(cirq.ops.X(qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.X(qubits[0]))
        circuit.append(cirq.ops.X(qubits[1]))
        circuit.append(cirq.ops.X(qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_Ygate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.Y(qubits[0]))
        circuit.append(cirq.ops.Y(qubits[1]))
        circuit.append(cirq.ops.Y(qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.Y(qubits[0]))
        circuit.append(cirq.ops.Y(qubits[1]))
        circuit.append(cirq.ops.Y(qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_Zgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.Z(qubits[0]))
        circuit.append(cirq.ops.Z(qubits[1]))
        circuit.append(cirq.ops.Z(qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.Z(qubits[0]))
        circuit.append(cirq.ops.Z(qubits[1]))
        circuit.append(cirq.ops.Z(qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_Hgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.H(qubits[0]))
        circuit.append(cirq.ops.H(qubits[1]))
        circuit.append(cirq.ops.H(qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.H(qubits[0]))
        circuit.append(cirq.ops.H(qubits[1]))
        circuit.append(cirq.ops.H(qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_Sgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.S(qubits[0]))
        circuit.append(cirq.ops.S(qubits[1]))
        circuit.append(cirq.ops.S(qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.S(qubits[0]))
        circuit.append(cirq.ops.S(qubits[1]))
        circuit.append(cirq.ops.S(qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_Tgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.T(qubits[0]))
        circuit.append(cirq.ops.T(qubits[1]))
        circuit.append(cirq.ops.T(qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.T(qubits[0]))
        circuit.append(cirq.ops.T(qubits[1]))
        circuit.append(cirq.ops.T(qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_RXgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.Rx(np.pi/5.5).on(qubits[0]))
        circuit.append(cirq.ops.Rx(np.pi/5.5).on(qubits[1]))
        circuit.append(cirq.ops.Rx(np.pi/5.5).on(qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.Rx(np.pi/5.5).on(qubits[0]))
        circuit.append(cirq.ops.Rx(np.pi/5.5).on(qubits[1]))
        circuit.append(cirq.ops.Rx(np.pi/5.5).on(qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_RYgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.Ry(np.pi/5.5).on(qubits[0]))
        circuit.append(cirq.ops.Ry(np.pi/5.5).on(qubits[1]))
        circuit.append(cirq.ops.Ry(np.pi/5.5).on(qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.Ry(np.pi/5.5).on(qubits[0]))
        circuit.append(cirq.ops.Ry(np.pi/5.5).on(qubits[1]))
        circuit.append(cirq.ops.Ry(np.pi/5.5).on(qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_RZgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.Rz(np.pi/5.5).on(qubits[0]))
        circuit.append(cirq.ops.Rz(np.pi/5.5).on(qubits[1]))
        circuit.append(cirq.ops.Rz(np.pi/5.5).on(qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.Rz(np.pi/5.5).on(qubits[0]))
        circuit.append(cirq.ops.Rz(np.pi/5.5).on(qubits[1]))
        circuit.append(cirq.ops.Rz(np.pi/5.5).on(qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_Ugate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.circuits.qasm_output.QasmUGate(0.1, 0.2, 0.3).on(qubits[0]))
        circuit.append(cirq.circuits.qasm_output.QasmUGate(0.4, 0.5, 0.6).on(qubits[1]))
        circuit.append(cirq.circuits.qasm_output.QasmUGate(0.7, 0.8, 0.9).on(qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.circuits.qasm_output.QasmUGate(0.1, 0.2, 0.3).on(qubits[0]))
        circuit.append(cirq.circuits.qasm_output.QasmUGate(0.4, 0.5, 0.6).on(qubits[1]))
        circuit.append(cirq.circuits.qasm_output.QasmUGate(0.7, 0.8, 0.9).on(qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_CNOTgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.CNOT(qubits[0], qubits[1]))
        circuit.append(cirq.ops.CNOT(qubits[1], qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.CNOT(qubits[0], qubits[1]))
        circuit.append(cirq.ops.CNOT(qubits[1], qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_CZgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.CZ(qubits[0], qubits[1]))
        circuit.append(cirq.ops.CZ(qubits[1], qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.CZ(qubits[0], qubits[1]))
        circuit.append(cirq.ops.CZ(qubits[1], qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_SWAPgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.SWAP(qubits[0], qubits[1]))
        circuit.append(cirq.ops.SWAP(qubits[1], qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.SWAP(qubits[0], qubits[1]))
        circuit.append(cirq.ops.SWAP(qubits[1], qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_XXgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.XX(qubits[0], qubits[1]))
        circuit.append(cirq.ops.XX(qubits[1], qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.XX(qubits[0], qubits[1]))
        circuit.append(cirq.ops.XX(qubits[1], qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_YYgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.YY(qubits[0], qubits[1]))
        circuit.append(cirq.ops.YY(qubits[1], qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.YY(qubits[0], qubits[1]))
        circuit.append(cirq.ops.YY(qubits[1], qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_ZZgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.ZZ(qubits[0], qubits[1]))
        circuit.append(cirq.ops.ZZ(qubits[1], qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.ZZ(qubits[0], qubits[1]))
        circuit.append(cirq.ops.ZZ(qubits[1], qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_CCXgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.CCX(qubits[0], qubits[1], qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.CCX(qubits[0], qubits[1], qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_CCZgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.CCZ(qubits[0], qubits[1], qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.CCZ(qubits[0], qubits[1], qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_TOFFOLIgate(self):
        """
        """
       
        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.TOFFOLI(qubits[0], qubits[1], qubits[2]))
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        circuit.append(cirq.ops.TOFFOLI(qubits[0], qubits[1], qubits[2]))
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        assert_allclose(actual, expected, rtol=1e-5, atol=0)


    def test_QulacsDensityMatrixSimulator_QuantumVolume(self):
        """
        """

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        parse_qasm_to_QulacsCircuit('tests/quantum_volume_n20_d8_0_0.qasm', circuit, qubits, qubit_n)
        qulacs_result = QulacsDensityMatrixSimulator().simulate(circuit)
        actual = qulacs_result.final_density_matrix

        qubit_n = 3
        qubits = [cirq.LineQubit(i) for i in range(qubit_n)]
        circuit = cirq.Circuit()
        parse_qasm_to_QulacsCircuit('tests/quantum_volume_n20_d8_0_0.qasm', circuit, qubits, qubit_n)
        cirq_result = cirq.DensityMatrixSimulator().simulate(circuit)
        expected = cirq_result.final_density_matrix

        print(actual-expected)
        assert_allclose(actual, expected, rtol=1e-2, atol=0)



if __name__ == "__main__":
    unittest.main()
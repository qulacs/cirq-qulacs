import matplotlib.pyplot as plt
import numpy as np
import time
import random
import re

import cirq
from cirqqulacs import QulacsSimulator


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

def main():
   bench_result_quantum_volume = [[],[]]

   with open('benchmark_gpu.csv', 'w') as f:
       f.write('nqubits,elapsed_time\n')
       for nqubits in range(5, 20+1):
           qubits = [cirq.LineQubit(i) for i in range(nqubits)]
           circuit = cirq.Circuit()
           parse_qasm_to_QulacsCircuit('quantum_volume_n{}_d8_0_0.qasm'.format(nqubits) ,circuit, qubits)

           start = time.time()
           qulacs_result = QulacsSimulator().simulate(circuit)
           elapsed_time = time.time() - start
           f.write('{},{}\n'.format(nqubits, elapsed_time))


if __name__ == '__main__':
   main()

import matplotlib.pyplot as plt
import numpy as np
import time
import random
from qulacs import QuantumState, QuantumStateGpu
from qulacs import QuantumCircuit
from qulacs.gate import DenseMatrix
from qulacs.circuit import QuantumCircuitOptimizer
from qulacs import QuantumState
from qulacs.gate import Identity, X,Y,Z
from qulacs.gate import H,S,Sdag, sqrtX,sqrtXdag,sqrtY,sqrtYdag
from qulacs.gate import T,Tdag
from qulacs.gate import RX,RY,RZ
from qulacs.gate import CNOT, CZ, SWAP
from qulacs.gate import U1,U2,U3

import re



def parse_qasm_to_QulacsCircuit(input_filename,qulacs_circuit):

   with open(input_filename, 'r') as ifile:
       lines = ifile.readlines()

       for line in lines:
           s = re.search(r"qreg|cx|u3|u1", line)

           if s is None:
               continue

           elif s.group() == 'qreg':
               match = re.search(r'\d\d*', line)
               # print(match)
               continue

           elif s.group() == 'cx':
               match = re.findall(r'\[\d\d*\]', line)
               c_qbit = int(match[0].strip('[]'))
               t_qbit = int(match[1].strip('[]'))
               qulacs_circuit.add_gate(CNOT(c_qbit,t_qbit))
               continue

           elif s.group() == 'u3':
               m_r = re.findall(r'[-]?\d\.\d\d*', line)
               m_i = re.findall(r'\[\d\d*\]', line)

               qulacs_circuit.add_gate(U3(int(m_i[0].strip('[]')),float(m_r[0]),float(m_r[1]),float(m_r[2])))

               continue

           elif s.group() == 'u1':
               m_r = re.findall(r'[-]?\d\.\d\d*', line)
               m_i = re.findall(r'\[\d\d*\]', line)

               qulacs_circuit.add_gate(U1(int(m_i[0].strip('[]')),float(m_r[0])))

               continue

def main():
   bench_result_quantum_volume = [[],[]]

   with open('benchmark_qulcas.csv', 'w') as f:
       f.write('nqubits,elapsed_time\n')
       for nqubits in range(5, 20+1):
           state = QuantumStateGpu(nqubits)
           state.set_zero_state()
           circuit = QuantumCircuit(nqubits)
           parse_qasm_to_QulacsCircuit('quantum_volume_n{}_d8_0_0.qasm'.format(nqubits) ,circuit)

           state.set_zero_state()
           start = time.time()
           circuit.update_quantum_state(state)
           elapsed_time = time.time() - start
           f.write('{},{}\n'.format(nqubits, elapsed_time))



if __name__ == '__main__':
   main()

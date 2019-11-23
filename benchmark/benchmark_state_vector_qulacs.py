import os
import re
import time
import random
import matplotlib.pyplot as plt
import numpy as np

from qulacs import QuantumState, QuantumStateGpu, QuantumCircuit

def parse_qasm_to_QulacsCircuit(input_filename, qulacs_circuit):

   with open(input_filename, "r") as ifile:
       lines = ifile.readlines()

       for line in lines:
           s = re.search(r"qreg|cx|u3|u1", line)

           if s is None:
               continue

           elif s.group() == 'qreg':
               match = re.search(r"\d\d*", line)
               continue

           elif s.group() == 'cx':
               match = re.findall(r'\[\d\d*\]', line)
               c_qbit = int(match[0].strip('[]'))
               t_qbit = int(match[1].strip('[]'))
               qulacs_circuit.add_CNOT_gate(c_qbit, t_qbit)
               continue

           elif s.group() == 'u3':
               m_r = re.findall(r'[-]?\d\.\d\d*', line)
               m_i = re.findall(r'\[\d\d*\]', line)
               t_qbit = int(m_i[0].strip('[]'))
               qulacs_circuit.add_U3_gate(t_qbit, float(m_r[0]),float(m_r[1]),float(m_r[2]))
               continue

           elif s.group() == 'u1':
               m_r = re.findall(r'[-]?\d\.\d\d*', line)
               m_i = re.findall(r'\[\d\d*\]', line) 
               t_qbit = int(m_i[0].strip('[]'))
               qulacs_circuit.add_U1_gate(t_qbit, float(m_r[0]))
               continue

def bench(state, circuit, mintime = 1.0):
    st = time.time()
    rep = 0
    while (time.time()-st) < mintime:
        circuit.update_quantum_state(state)
        rep += 1
    elp = (time.time()-st)/rep
    return elp

def bench_sweep(QuantumStateClass, bench_name, folder_path = "./result/"):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    fname = "_".join( ["benchmark", "state_vector", bench_name] ) +".csv"
    fout = open(folder_path+fname, 'w')
    fout.write('n_qubits,n_iter,elapsed_time\n')
    fout.close()
    for niter in range(10):
        for nqubits in range(5, 25+1):
            circuit = QuantumCircuit(nqubits)
            state = QuantumStateClass(nqubits)
            parse_qasm_to_QulacsCircuit('quantum_volume/quantum_volume_n{}_d8_0_{}.qasm'.format(nqubits, niter) ,circuit)

            elapsed_time = bench(state, circuit)

            fout = open(folder_path+fname, 'a')
            fout.write('{},{},{}\n'.format(nqubits, niter, elapsed_time))
            fout.close()
            print(bench_name + '{},{},{}'.format(niter, nqubits, elapsed_time))
            del state


def main():
    bench_sweep(QuantumState, "qulacs_cpu", "./result_qulacs/")
    bench_sweep(QuantumStateGpu, "qulacs_gpu", "./result_qulacs/")

if __name__ == '__main__':
   main()

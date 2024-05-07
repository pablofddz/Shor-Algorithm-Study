# Shor Algorithm Study
Code used to study Shor's algorithm and its implementations
Base code taken from https://github.com/sundewang233/shor_algorithm_simulation 

# Packages needed
Qiskit 1.0.2 and Qiskit-Aer-Gpu 0.14.01
running on Python 3.10.12

# Files
1.- onquantum.py: File that runs Shor's algorithm on IBM's Brisbane Quantum Computer
2.- original.py: File that runs Shor's algorithm semi-clasically with the version found on the original GitHub
                 but for the new Qiskit version's updates and for the way of analysing the results
3.- parallelized.py: File that runs Shor's algorithm semi-clasically with the version found on the original GitHub
                 but parallelizing the construction of the circuit (doesn't improve time) and 
                 with the new Qiskit version's updates and an improved way of analysing the results
4.- mejorada.py: File that runs Shor's algorithm semi-clasically with the version found on the original GitHub
                 but reducing considerably the number of gates using an idea from https://arxiv.org/pdf/quant-ph/0001066 and 
                 with the new Qiskit version's updates and an improved way of analysing the results

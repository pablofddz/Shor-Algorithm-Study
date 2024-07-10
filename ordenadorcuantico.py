

""" Imports from qiskit"""
from concurrent.futures import ProcessPoolExecutor
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_ibm_provider import IBMProvider # type: ignore


from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import SamplerV2 as Sampler

""" Imports to Python functions """
import math
import numpy as np
import contfrac

""" Function to check if N is of type q^p"""
def check_if_power(N):
    """ Check if N is a perfect power in O(n^3) time, n=ceil(logN) """
    b=2
    while (2**b) <= N:
        a = 1
        c = N
        while (c-a) >= 2:
            m = int( (a+c)/2 )

            if (m**b) < (N+1):
                p = int( (m**b) )
            else:
                p = int(N+1)

            if int(p) == int(N):
                print('N is {0}^{1}'.format(int(m),int(b)) )
                return True

            if p<N:
                a = int(m)
            else:
                c = int(m)
        b=b+1

    return False

def get_factors_original(l_phi,N,a,prec):
    # construct decimal value of phi
    n = 0
    phi_tilde = 0
    for l in l_phi:
        n -= 1
        phi_tilde = phi_tilde + 2**n * int(l)
    # express phi_tilde as a fraction
    res = len(str(phi_tilde)) - 2 # subtract 2 for "0."
    scale = 10**res # automated scale set by res
    num = int(phi_tilde*scale)
    den = int(scale)
    # in lowest terms
    c = math.gcd(num, den)
    num = int(num / c)
    den = int(den / c)
    phi = (num, den)

    # construct convergents for phi
    convergents = list(contfrac.convergents(phi, prec))

    # check convergents for solution
    for conv in convergents:
        r = conv[1]
        if r>100000:
            break
        test1 = r % 2 # 0 if r is even
        x = a**int(r/2)
        test2 = (x-1) % N # 0 if a^r/2 is a trivial root
        test3 = (x+1) % N # 0 if a^r/2 is a trivial root
        test4 = a**r % N # 1 if r is a solution
        if (test1==0 and test2!=0 and test3!=0 and test4==1):
            print(N," = ", math.gcd(x-1, N), " x ", math.gcd(x+1, N))
            return True
    
    print("No se halló la factorización")
    
    return False

def get_factors_new(l_phi,N,a,prec):
    # construct decimal value of phi
    n = 0
    phi_tilde = 0
    for l in l_phi:
        n -= 1
        phi_tilde = phi_tilde + 2**n * int(l)
    res = len(str(phi_tilde)) - 2 # subtract 2 for "0."
    scale = 10**res # automated scale set by res
    num = int(phi_tilde*scale)
    den = int(scale)
    # in lowest terms
    c = math.gcd(num, den)
    num = int(num / c)
    den = int(den / c)
    phi = (num, den)

    # construct convergents for phi
    convergents = list(contfrac.convergents(phi, prec))

    # check convergents for solution
    for conv in convergents:
        r = conv[1]
        if r>100000:
            break
        while r % 2 == 0 and a**r % N == 1:
            x = a**int(r/2)
            if x % N == 1:
                r = r/2
            elif x % N == N-1:
                print("Se ha hallado el periodo, pero con este valor de a no se puede hallar la factorización")
                break
            else:
                print(N, " = ", math.gcd(x-1, N), " x ", math.gcd(x+1, N))
                return True

    print("No se halló la factorización")
    return False

def process_result(i, sim_result, number_shots, N, a, analisis):
    """Process a single result from simulation."""
    output_desired = list(sim_result.keys())[i]
    
    prob_this_result = 100 * (int(list(sim_result.values())[i])) / number_shots

    if analisis == 0:
        success = get_factors_original(output_desired, int(N), int(a), 10000)
    else:
        success = get_factors_new(output_desired, int(N), int(a), 10000)
    return success, prob_this_result

"""Functions that calculate the modular inverse using Euclid's algorithm"""
def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)
def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

""" Function to create the QFT """
def create_QFT(circuit,up_reg,n,with_swaps, k):
    i=n-1
    """ Apply the H gates and Cphases"""
    """ The Cphases with |angle| < threshold are not created because they do 
    nothing. The threshold is put as being 0 so all CPhases are created,
    but the clause is there so if wanted just need to change the 0 of the
    if-clause to the desired value """
    
    while i>=0:
        circuit.h(up_reg[i])        
        j=i-1  
        while j>=0 and (i - j<= k or k==0):
            if (np.pi)/(pow(2,(i-j))) > 0:
                circuit.cp( (np.pi)/(pow(2,(i-j))) , up_reg[i] , up_reg[j] )
                j=j-1   
        i=i-1  

    """ If specified, apply the Swaps at the end """
    if with_swaps==1:
        i=0
        while i < ((n-1)/2):
            circuit.swap(up_reg[i], up_reg[n-1-i])
            i=i+1

""" Function to create inverse QFT """
def create_inverse_QFT(circuit,up_reg,n,with_swaps, k):
    """ If specified, apply the Swaps at the beggining"""
    if with_swaps==1:
        i=0
        while i < ((n-1)/2):
            circuit.swap(up_reg[i], up_reg[n-1-i])
            i=i+1
    
    """ Apply the H gates and Cphases"""
    """ The Cphases with |angle| < threshold are not created because they do 
    nothing. The threshold is put as being 0 so all CPhases are created,
    but the clause is there so if wanted just need to change the 0 of the
    if-clause to the desired value """
    i=0
    while i<n:
        circuit.h(up_reg[i])
        if i != n-1:
            j=i+1
            y=i
            while y>=0 and (j - y <= k or k==0):
                 if (np.pi)/(pow(2,(j-y))) > 0:
                    circuit.cp( - (np.pi)/(pow(2,(j-y))) , up_reg[j] , up_reg[y] )
                    y=y-1   
        i=i+1

"""Function that calculates the angle of a phase shift in the sequential QFT based on the binary digits of a."""
"""a represents a possile value of the classical register"""
def getAngle(a, N):
    """convert the number a to a binary string with length N"""
    s=bin(int(a))[2:].zfill(N) 
    angle = 0
    for i in range(0, N):
        """if the digit is 1, add the corresponding value to the angle"""
        if s[N-1-i] == '1': 
            angle += math.pow(2, -(N-i))
    angle *= np.pi
    return angle

"""Function that calculates the array of angles to be used in the addition in Fourier Space"""
def getAngles(a,N):
    s=bin(int(a))[2:].zfill(N) 
    angles=np.zeros([N])
    for i in range(0, N):
        for j in range(i,N):
            if s[j]=='1':
                angles[N-i-1]+=math.pow(2, -(j-i))
        angles[N-i-1]*=np.pi
    return angles

"""Creation of a doubly controlled phase gate"""
def ccphase(circuit, angle, ctl1, ctl2, tgt):
    circuit.cp(angle/2,ctl1,tgt)
    circuit.cx(ctl2,ctl1)
    circuit.cp(-angle/2,ctl1,tgt)
    circuit.cx(ctl2,ctl1)
    circuit.cp(angle/2,ctl2,tgt)

"""Creation of the circuit that performs addition by a in Fourier Space"""
"""Can also be used for subtraction by setting the parameter inv to a value different from 0"""
def phiADD(circuit, q, a, N, inv):
    angle=getAngles(a,N)
    for i in range(0,N):
        if inv==0:
            circuit.p(angle[i],q[i])
            """addition"""
        else:
            circuit.p(-angle[i],q[i])
            """subtraction"""

"""Single controlled version of the phiADD circuit"""
def cphiADD(circuit, q, ctl, a, n, inv):
    angle=getAngles(a,n)
    for i in range(0,n):
        if inv==0:
            circuit.cp(angle[i],ctl,q[i])
        else:
            circuit.cp(-angle[i],ctl,q[i])
        
"""Doubly controlled version of the phiADD circuit"""
def ccphiADD(circuit,q,ctl1,ctl2,a,n,inv):
    angle=getAngles(a,n)
    for i in range(0,n):
        if inv==0:
            ccphase(circuit,angle[i],ctl1,ctl2,q[i])
        else:
            ccphase(circuit,-angle[i],ctl1,ctl2,q[i])
        
"""Circuit that implements doubly controlled modular addition by a"""
def ccphiADDmodN(circuit, q, ctl1, ctl2, aux, a, N, n, kmax):
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 0)
    phiADD(circuit, q, N, n, 1)
    create_inverse_QFT(circuit, q, n, 0, kmax)
    circuit.cx(q[n-1],aux)
    create_QFT(circuit,q,n,0, kmax)
    cphiADD(circuit, q, aux, N, n, 0)
    
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 1)
    create_inverse_QFT(circuit, q, n, 0, kmax)
    circuit.x(q[n-1])
    circuit.cx(q[n-1], aux)
    circuit.x(q[n-1])
    create_QFT(circuit,q,n,0, kmax)
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 0)

"""Circuit that implements the inverse of doubly controlled modular addition by a"""
def ccphiADDmodN_inv(circuit, q, ctl1, ctl2, aux, a, N, n, kmax):
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 1)
    create_inverse_QFT(circuit, q, n, 0, kmax)
    circuit.x(q[n-1])
    circuit.cx(q[n-1],aux)
    circuit.x(q[n-1])
    create_QFT(circuit, q, n, 0, kmax)
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 0)
    cphiADD(circuit, q, aux, N, n, 1)
    create_inverse_QFT(circuit, q, n, 0, kmax)
    circuit.cx(q[n-1], aux)
    create_QFT(circuit, q, n, 0, kmax)
    phiADD(circuit, q, N, n, 0)
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 1)

"""Circuit that implements single controlled modular multiplication by a"""
def cMULTmodN(circuit, ctl, q, aux, a, N, n, kmax):
    create_QFT(circuit,aux,n+1,0, kmax)
    for i in range(0, n):
        ccphiADDmodN(circuit, aux, q[i], ctl, aux[n+1], (2**i)*a % N, N, n+1, kmax)
    create_inverse_QFT(circuit, aux, n+1, 0, kmax)

    for i in range(0, n):
        circuit.cswap(ctl,q[i],aux[i])

    a_inv = modinv(a, N)
    create_QFT(circuit, aux, n+1, 0, kmax)
    i = n-1
    while i >= 0:
        ccphiADDmodN_inv(circuit, aux, q[i], ctl, aux[n+1], math.pow(2,i)*a_inv % N, N, n+1, kmax)
        i -= 1
    create_inverse_QFT(circuit, aux, n+1, 0, kmax)

""" Main program """
if __name__ == '__main__':

    """ Ask for analysis number N """   

    N = int(input('Escribe el número N natural que quieres factorizar: '))
    
    """ Check if N==1 or N==0"""

    if N==1 or N==0: 
       print('Inserta un número distinto de 1 y 0')
       exit()
    
    """ Check if N is even """

    if (N%2)==0:
        print('N es par')
        exit()
    
    """ Check if N can be put in N=p^q, p>1, q>=2 """

    """ Try all numbers for p: from 2 to sqrt(N) """
    if check_if_power(N)==True:
       exit()

    print('Es necesario usar el algoritmo de Shor\n')

    """ Get n value used in Shor's algorithm, to know how many qubits are used """
    n = math.ceil(math.log(N,2))

    print("Predeterminado: \n\t a=2, \n\t 2n qubits de precisión, \n\t transformada cuántica de Fourier aproximada, \n\t análisis de resultados refinado y paralelo.\n")
    personalizado = int(input('¿Quieres personalizar los ajustes del algoritmo o deseas usar los predeterminados? (0-Personalizar, 1-Predeterminados): '))
    if personalizado != 0 and personalizado != 1:
        print('Selecciona una opción correcta')
        exit()
    elif personalizado == 0:
        a = int(input('Inserta el número a (a=2 es lo que se usó en los experimentos): '))
        if a<1:
            print('Selecciona un valor correcto')
            exit()

        x = math.gcd(a,N)
        if x != 1:
            print("a tiene el factor ", x, " en común con N")
            exit()

        precision = int(input('¿Quieres usar 2n qubits de precisión? (0 - No, 1 - Sí) : '))
        if precision != 0 and precision != 1:
            print('Selecciona una opción correcta')
            exit()

        if precision == 0:
            precision = int(input('Escriba el número de qubits de precisión que quieres utilizar: '))
            if precision < 1:
                print("No es posible usar ese número de qubits de precisión")
                exit()
        else:
            precision = 2*n

        qubits = precision + 2*n + 2

        approximate=int(input('¿Usar la versión aproximada de la transformada cuántica de Fourier? (0 - No, 1 - Sí): '))
        if approximate == 1:
            kmax = math.ceil(math.log(n, 2))
        elif approximate == 0:
            kmax = 0
        else:
            print('Selecciona una opción correcta')
            exit()

        analisis=int(input('¿Usar la versión refinada de análisis de resultados? (0 - No, 1 - Sí): '))
        if analisis != 0 and analisis != 1:
            print('Selecciona una opción correcta')
            exit()
        
        paralelo=int(input('¿Usar análisis en paralelo de resultados? (0 - No, 1 - Sí): ')) 
        if paralelo != 0 and paralelo != 1:
            print('Selecciona una opción correcta')
            exit()
    else:
        a = 2

        precision = 2*n
        
        qubits = 2*n + 2 + precision

        approximate=1
        if approximate == 1:
            kmax = math.ceil(math.log(n, 2))
        elif approximate == 0:
            kmax = 0
        else:
            print('Selecciona una opción correcta')
            exit()

        analisis=1
        
        paralelo=1

    print("\nEn total se van a usar", qubits, "qubits")
    print("\n¡Tener en cuenta que si el número de qubits es mayor que 25 es muy posible que el ordenador cuántico devuelva un error!\n")

    """ Create quantum and classical registers """
    import time
    start = time.time()
    """auxilliary quantum register used in addition and multiplication"""
    aux = QuantumRegister(n+2)
    """quantum register where the sequential QFT is performed"""
    up_reg = QuantumRegister(precision)
    """quantum register where the multiplications are made"""
    down_reg = QuantumRegister(n)
    """classical register where the measured values of the QFT are stored"""
    up_classic = ClassicalRegister(precision)

    """ Create Quantum Circuit """
    circuit = QuantumCircuit(down_reg , up_reg , aux, up_classic)

    """ Initialize down register to 1 and create maximal superposition in top register """
    circuit.h(up_reg)
    circuit.x(down_reg[0])

    """ Apply the multiplication gates as showed in the report in order to create the exponentiation """
    for i in range(0, precision):
        cMULTmodN(circuit, up_reg[i], down_reg, aux, int(pow(a, pow(2, i))), N, n, kmax)

    """ Apply inverse QFT """
    create_inverse_QFT(circuit, up_reg, precision,1, kmax)

    """ Measure the top qubits, to get x value"""
    circuit.measure(up_reg,up_classic)

    end = time.time()
    print("\nCircuito construido")
    tiempo_circuito = end - start
    print('Tiempo necesitado: %f segundos'%(tiempo_circuito))

    """ Select how many times the circuit runs"""
    number_shots=int(input('Número de intentos: '))
    if number_shots < 1:
        print('Al menos una vez...')
        exit()

    """ Print info to user """
    start = time.time()
    print('Ejecutando el circuito {0} veces para N={1} y a={2}\n'.format(number_shots,N,a))

    
    """ Simulate the created Quantum Circuit """  
    service = QiskitRuntimeService(channel='ibm_quantum')
    provider = IBMProvider()
    backend = provider.backends(name="ibm_brisbane")[0]
    session = Session(service=service, backend="ibm_brisbane")

    transpiled_circuit = transpile(circuit, backend=backend)
    print("Circuito transpilado")

    result = Sampler(session=session).run([transpiled_circuit], shots=number_shots).result()
    counts_result = result[0].data.c0.get_counts()

    end = time.time()
    print("Ejecución completada")
    tiempo_ejecucion = end-start
    print('Tiempo necesitado: %f segundos'%(tiempo_ejecucion))

    while i < len(counts_result):
        print('El resultado \"{0}({1})\" ocurrió {2} veces de {3}'.format(list(counts_result.keys())[i], int(list(counts_result.keys())[i], 2), list(counts_result.values())[i],number_shots))
        i=i+1

    """ An empty print just to have a good display in terminal """
    print(' ')

    start = time.time()
    """ Initialize this variable """
    prob_success=0
    
    """ For each simulation result, print proper info to user and try to calculate the factors of N"""
    if paralelo == 1:
        i=0
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_result, i, counts_result, number_shots, N, a, analisis) for i in range(len(counts_result))]
            for future in futures:
                success, prob_this_result = future.result()
                if success:
                    prob_success += prob_this_result
    else:
        for i in range(len(counts_result)):
            success, prob_this_result = process_result(i, counts_result, number_shots, N, a, analisis)
            if success:
                prob_success += prob_this_result
            i=i+1


    end = time.time()
    print('Tiempo necesitado para el análisis de resultados %f'%(end - start))

    print("\nUsando a={0}, se han encontrado los factores de N={1} en un {2:.4f} % de los casos\n".format(a,N,prob_success))
    print("Tiempo de construcción del circuito: %f segundos"%(tiempo_circuito))
    print("Tiempo de ejecución del circuito: %f segundos"%(tiempo_ejecucion))
    print("Tiempo de análisis de resultados: %f segundos"%(end-start))
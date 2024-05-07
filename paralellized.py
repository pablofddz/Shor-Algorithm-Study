"""
This is the final implementation of Shor's Algorithm using the circuit presented in section 2.3 of the report about the second
simplification introduced by the base paper used.
The circuit is general, so, in a good computer that can support simulations infinite qubits, it can factorize any number N. The only limitation
is the capacity of the computer when running in local simulator and the limits on the IBM simulator (in the number of qubits and in the number
of QASM instructions the simulations can have when sent to IBM simulator).
The user may try N=21, which is an example that runs perfectly fine even just in local simulator because, as in explained in report, this circuit,
because implements the QFT sequentially, uses less qubits then when using a "normal"n QFT.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

""" Imports from qiskit"""
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import AerSimulator, QasmSimulator

""" Imports to Python functions """
import math
import array
import fractions
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

""" Function to get the value a ( 1<a<N ), such that a and N are coprime. Starts by getting the smallest a possible
    This normally can be done fully randomly, we just did it like this for thr user (professor) to have control 
    over the a value that gets selected """
def get_value_a(N):

    """ ok defines if user wants to used the suggested a (if ok!='0') or not (if ok=='0') """
    ok='0'

    """ Starting with a=2 """
    a=2

    """ Get the smallest a such that a and N are coprime"""
    while math.gcd(a,N)!=1:
        a=a+1

    """ Store it as the smallest a possible """
    smallest_a = a

    """ Ask user if the a found is ok, if not, then increment and find the next possibility """  
    ok = input('Is the number {0} ok for a? Press 0 if not, other number if yes: '.format(a))
    if ok=='0':
        if(N==3):
            print('Number {0} is the only one you can use. Using {1} as value for a\n'.format(a,a))
            return a
        a=a+1

    """ Cycle to find all possibilities for a not counting the smallest one, until user says one of them is ok """
    while ok=='0':
        
        """ Get a coprime with N """
        while math.gcd(a,N)!=1:
            a=a+1
    
        """ Ask user if ok """
        ok = input('Is the number {0} ok for a? Press 0 if not, other number if yes: '.format(a))

        """ If user says it is ok, then exit cycle, a has been found """
        if ok!='0':
            break
        
        """ If user says it is not ok, increment a and check if are all possibilites checked.  """
        a=a+1

        """ If all possibilities for a are rejected, put a as the smallest possible value and exit cycle """
        if a>(N-1):
            print('You rejected all options for value a, selecting the smallest one\n')
            a=smallest_a
            break

    """ Print the value that is used as a """
    print('Using {0} as value for a\n'.format(a))

    return a

def get_factors(l_phi,N,a,prec):
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
    print("convergents of phi:", convergents)

    # check convergents for solution
    for conv in convergents:
        r = conv[1]
        test1 = r % 2 # 0 if r is even
        x = a**int(r/2)
        test2 = (x-1) % N # 0 if a^r/2 is a trivial root
        test3 = (x+1) % N # 0 if a^r/2 is a trivial root
        test4 = a**r % N # 1 if r is a solution
        if (test1==0 and test2!=0 and test3!=0 and test4==1):
            print("conv:", conv, "r =", r, ": factors")
            print("factor1:", math.gcd(x-1, N))
            print("factor2:", math.gcd(x+1, N))
            return True
    
    print("no factors found")
    
    return False

""" Function to apply the continued fractions to find r and the gcd to find the desired factors"""
def get_factors_old(x_value,t_upper,N,a, prec):
    #print(x_value)
    if x_value<=0:
        # print('x_value is <= 0, there are no continued fractions\n')
        return False

    # print('Running continued fractions for this case\n')

    """ Calculate T and x/T """
    T = pow(2,t_upper)

    x_over_T = x_value/T

    """ Cycle in which each iteration corresponds to putting one more term in the
    calculation of the Continued Fraction (CF) of x/T """

    """ Initialize the first values according to CF rule """
    i=0
    b = array.array('i')
    t = array.array('f')

    b.append(math.floor(x_over_T))
    t.append(x_over_T - b[i])

    while i>=0:

        """From the 2nd iteration onwards, calculate the new terms of the CF based
        on the previous terms as the rule suggests"""

        if i>0:
            #print(x_value, N, a)
            if t[i-1] == 0:
                return
            b.append( math.floor( 1 / (t[i-1]) ) ) 
            
            t.append( ( 1 / (t[i-1]) ) - b[i] )

        """ Calculate the CF using the known terms """

        aux = 0
        j=i
        while j>0:    
            aux = 1 / ( b[j] + aux )      
            j = j-1
        
        aux = aux + b[0]

        """Get the denominator from the value obtained"""
        frac = fractions.Fraction(aux).limit_denominator(N)
        den=frac.denominator
        # print('Approximation number {0} of continued fractions:'.format(i+1))
        # print("Numerator:{0} \t\t Denominator: {1}\n".format(frac.numerator,frac.denominator))

        """ Increment i for next iteration """
        i=i+1

        if (den%2) == 1:
            if i>=prec:
                print('Returning because have already done too much tries')
                return False
            # print('Odd denominator, will try next iteration of continued fractions\n')
            continue
    
        """ If denominator even, try to get factors of N """

        """ Get the exponential a^(r/2) """

        exponential = 0

        if den<1000:
            a = int(a)
            exp = int(den/2)
            exponential=pow(a , exp)
        
        """ Check if the value is too big or not """
        #if exponential>1000000000000:
            #print('Denominator of continued fraction is too big!\n')
            # aux_out = input('Input number 1 if you want to continue searching, other if you do not: ')
            # if aux_out != '1':
            #return False
            # else:
            #     continue
            # continue

        """If the value is not to big (infinity), then get the right values and
        do the proper gcd()"""

        putting_plus = int(exponential + 1)

        putting_minus = int(exponential - 1)
    
        one_factor = math.gcd(putting_plus,N)
        other_factor = math.gcd(putting_minus,N)
    
        """ Check if the factors found are trivial factors or are the desired
        factors """
        #print(one_factor)
        if (one_factor==1 or one_factor==N) or (other_factor==1 or other_factor==N):
            #print('Found just trivial factors, not good enough\n')
            """ Check if the number has already been found, use i-1 because i was already incremented """
            # print(i, t[i - 1])
            if t[i-1]==0:
                print('The continued fractions found exactly x_final/(2^(2n)) , leaving funtion\n')
                return False
            if i<prec:
                # aux_out = input('Input number 1 if you want to continue searching, other if you do not: ')
                # if aux_out != '1':
                #     return False       
                pass
            else:
                """ Return if already too much tries and numbers are huge """ 
                print('Returning because have already done too many tries\n')
                return False         
        else:
            print('The factors of {0} are {1} and {2}\n'.format(N,one_factor,other_factor))
            print('Found the desired factors!\n')
            return True

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

def show_good_coef(results, n):
    i=0
    max = pow(2,n)
    """ Iterate to all possible states """
    while i<max:
        binary = bin(i)[2:].zfill(n)
        number = results.item(i)
        number = round(number.real, 3) + round(number.imag, 3) * 1j
        """ Print the respective component of the state if it has a non-zero coeficient """
        if number!=0:
            print('|{}>'.format(binary),number)
        i=i+1

def process_result(i, sim_result, number_shots, n, N, a):
    """Process a single result from simulation."""
    all_registers_output = list(sim_result.get_counts().keys())[i]
    output_desired = all_registers_output.split(" ")[1]
    x_value = int(output_desired, 2)
    prob_this_result = 100 * (int(list(sim_result.get_counts().values())[i])) / number_shots

    print("------> Analysing result {0}. This result happened in {1:.4f} % of all cases\n".format(output_desired, prob_this_result))
    print('In decimal, x_final value for this result is: {0}\n'.format(x_value))

    success = get_factors(output_desired, int(N), int(a), 10000)
    #success = get_factors_old(x_value, int(2*n), int(N), int(a), 10000)
    return success, prob_this_result

def create_circuit(n,i,a,N,kmax):
    """auxilliary quantum register used in addition and multiplication"""
    aux = QuantumRegister(n+2)
    """single qubit where the sequential QFT is performed"""
    up_reg = QuantumRegister(1)
    """quantum register where the multiplications are made"""
    down_reg = QuantumRegister(n)
    """classical register where the measured values of the sequential QFT are stored"""
    up_classic = ClassicalRegister(8*n)
    """classical bit used to reset the state of the top qubit to 0 if the previous measurement was 1"""
    c_aux = ClassicalRegister(1)
    """ Create Quantum Circuit """
    circuit = QuantumCircuit(down_reg, up_reg, aux, up_classic, c_aux)
    if i==0:
        circuit.x(down_reg[0])
    """reset the top qubit to 0 if the previous measurement was 1"""
    circuit.x(up_reg).c_if(c_aux, 1)
    circuit.h(up_reg)
    cMULTmodN(circuit, up_reg[0], down_reg, aux, a**(2**(8*n-1-i)), N, n, kmax)
    """cycle through all possible values of the classical register and apply the corresponding conditional phase shift"""
    for j in range(2, i+1):
        """the phase shift is applied if the value of the classical register matches j exactly"""
        circuit.p(-math.pi*2**(1-j), up_reg[0]).c_if(up_classic[i+1-j], 1)
    circuit.h(up_reg)
    circuit.measure(up_reg[0], up_classic[i])
    circuit.measure(up_reg[0], c_aux[0])

    return circuit

def compose_circuit_fragment(circuits):
    """compose all circuits of the fragment"""
    if not circuits:  
        return None 
    composed_circuit = circuits[0]
    for circuit in circuits[1:]:
        composed_circuit = composed_circuit.compose(circuit)
    return composed_circuit

def parallel_compose_circuits(circuits):
    """divide and compose circuits in parallel"""
    num_cpus = min(multiprocessing.cpu_count(), len(circuits)) 
    # divide in fragments
    if num_cpus == 0:  # at least a fragment
        return None 
    fragment_size = len(circuits) // num_cpus
    circuit_fragments = [circuits[i * fragment_size:(i + 1) * fragment_size] for i in range(num_cpus)]
    
    # make sure all circuits are included
    remainder = len(circuits) % num_cpus
    if remainder:
        last_fragment = circuits[-remainder:] 
        circuit_fragments.append(last_fragment)
    
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        composed_fragments = [executor.submit(compose_circuit_fragment, circuit_fragment) for circuit_fragment in circuit_fragments]
    
    verified_composed_fragments = [frag.result() for frag in composed_fragments if frag.result() is not None]
    # compose the circuits
    if composed_fragments:
        final_circuit = verified_composed_fragments[0]
        for composed_fragment in verified_composed_fragments[1:]:
            final_circuit = final_circuit.compose(composed_fragment)
        return final_circuit
    else:
        return None

""" Main program """
if __name__ == '__main__':

    """ Ask for analysis number N """   

    N = int(input('Please insert integer number N: '))
    
    print('input number was: {0}\n'.format(N))
    
    """ Check if N==1 or N==0"""

    if N==1 or N==0: 
       print('Please put an N different from 0 and from 1')
       exit()
    
    """ Check if N is even """

    if (N%2)==0:
        print('N is even, so does not make sense!')
        exit()
    
    """ Check if N can be put in N=p^q, p>1, q>=2 """

    """ Try all numbers for p: from 2 to sqrt(N) """
    if check_if_power(N)==True:
       exit()

    print('Not an easy case, using the quantum circuit is necessary\n')

    """ To login to IBM Q experience the following functions should be called """
    """
    IBMQ.delete_accounts()
    IBMQ.save_account('insert token here')
    IBMQ.load_accounts()"""

    """ Get an integer a that is coprime with N """
    # a = int(input('Please insert integer number a: '))
    a = 2
    """ If user wants to force some values, can do that here, please make sure to update print and that N and a are coprime"""
    """print('Forcing N=15 and a=4 because its the fastest case, please read top of source file for more info')
    N=15
    a=2"""

    """ Get n value used in Shor's algorithm, to know how many qubits are used """
    n = math.ceil(math.log(N,2))
    kmax = math.ceil(math.log(n, 2))
    # kmax = n
    
    print('Total number of qubits used: {0}\n'.format(2*n+3))
    """ Select if you want to run the approximate QFT version"""
    #approximate=int(input('Approximate QFT? (0 - No, 1 - Yes): '))
    approximate = 1
    if approximate == 1:
        kmax = math.ceil(math.log(n, 2))
    elif approximate == 0:
        kmax = 0
    else:
        print('Please select a correct option..')
        exit()

    import time
    start = time.time()
    
    """ Cycle to create the Sequential QFT, measuring qubits and applying the right gates according to measurements """
        
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(create_circuit, n,i,a,N,kmax) for i in range(0, 8*n)]

    circuits=[circuit.result() for circuit in futures]

    parallel_type = 1
    if parallel_type == 1:      
        circuit = parallel_compose_circuits(circuits)
    if parallel_type == 2:
        circuit = circuits[0].result()
        for frag in circuits[1:]:
            circ = frag.result()
            circuit.compose(circ, inplace=True)
    
    end = time.time()
    
    print('time used is %f'%(end - start))


    """ Select how many times the circuit runs"""
    number_shots=int(input('Number of times to run the circuit: '))
    if number_shots < 1:
        print('Please run the circuit at least one time...')
        exit()


    start = time.time()
    """ Print info to user """
    print('Executing the circuit {0} times for N={1} and a={2}\n'.format(number_shots,N,a))

    """ Simulate the created Quantum Circuit """
    # backend = QasmSimulator(method='statevector', max_parallel_threads = 56)
    
    simulator_gpu = AerSimulator(method='matrix_product_state', device='GPU')
    #simulator_gpu = QasmSimulator(method='statevector', device='GPU')
    # simulator_gpu = Aer.get_backend('aer_simulator', device = 'GPU')
    # simulator_gpu.set_options(device='GPU', method = 'statevector_gpu')
    # simulator_gpu.set_options(device='GPU', method = "statevector")
    
    transpiled_circuit = transpile(circuit, simulator_gpu)

    print("circuit transpiled")
    simulation = simulator_gpu.run(transpiled_circuit, shots=number_shots)
    sim_result = simulation.result()
    counts_result = sim_result.get_counts()

    """ Get the results of the simulation in proper structure """
    #sim_result=simulation.result()
    #counts_result = sim_result.get_counts(circuit)
    # import matplotlib.pyplot as plt
    # plot_histogram(counts_result)
    # plt.show()
    end = time.time()
    """ Print info to user from the simulation results """
    # show_good_coef(counts_result, 2 * n)
    print('Printing the various results followed by how many times they happened (out of the {} cases):\n'.format(number_shots))
    print(counts_result)
    i=0
    while i < len(counts_result):
        print('Result \"{0}({1})\" happened {2} times out of {3}'.format(list(counts_result.keys())[i], int(list(counts_result.keys())[i].split(' ')[1], 2), list(counts_result.values())[i],number_shots))
        i=i+1
    
    print('time used is %f'%(end - start))

    """ An empty print just to have a good display in terminal """
    print(' ')


    start = time.time()
    """ Initialize this variable """
    prob_success=0
    
    """ For each simulation result, print proper info to user and try to calculate the factors of N"""
    i=0
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_result, i, sim_result, number_shots, n, N, a) for i in range(len(counts_result))]
        for future in futures:
            success, prob_this_result = future.result()
            if success:
                prob_success += prob_this_result

    end = time.time()
    print('time used is %f'%(end - start))

    print("\nUsing a={0}, found the factors of N={1} in {2:.4f} % of the cases\n".format(a,N,prob_success))
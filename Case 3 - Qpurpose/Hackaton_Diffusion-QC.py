import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT, ModularAdderGate
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
# The block encoding is built from three special gates: shift, a special 2 qubit gate prep and a modular adder
def QFT_Shift_gate(n): #https://egrettathula.wordpress.com/2024/07/28/quantum-incrementer/
    """
    n : number of qubits.
    
    Returns a gate implementing the shift operation  
    |k> --> |k+1 (mod 2**n)> in the computational basis.
    """
    circuit = QuantumCircuit(n, name= 'QFTShift')
    circuit.compose(QFT(n, inverse=False, do_swaps=False, name='QFT'), range(n), inplace=True)
    for m in range(n):
        circuit.p(np.pi / (2 ** m), m)
    circuit.compose(QFT(n, inverse=True, do_swaps=False, name='QFT'), range(n), inplace=True)
    return circuit.to_gate()

def MCX_Shift_gate(n):
    """
    n : number of qubits.
    
    Returns a gate implementing the shift operation  
    |k> --> |k+1 (mod 2**n)> in the computational basis.
    """
    qc = QuantumCircuit(n,name = 'MCXShift')
    for j in range(n-1):
        qc.mcx([k for k in range(n-1-j)],n-1-j)         # Using multi controlled NOT gates 
    qc.x(0)
    return qc.to_gate()
def Diffusion_prep(n,dt,nu):
    """
    n: number of spatial qubits. N = 2**n is the number of spatial grid points. 
    dt: time step.      
    nu: diffusion coefficient. 
    
    Returns a 2 qubit gate state preparation gate implementing 
    |0> --> sqrt(nu*dt/dx**2)|0> + sqrt(1-2*nu*dt/dx**2)|1> + sqrt(nu*dt/dx**2) |2>
    """
    d = 4                               # The domain is fixed to be [0,4]
    dx = d/2**n                
    a = 1-2*dt*nu/(dx**2)
    if a>0:
        theta = np.arcsin(np.sqrt(a))   # The rotation angle needed to prepare a using an RY gate  
    else:
        print('The chosen values n,dt,nu are not admissible. Arrange that 1>2*nu*dt/(dx^2)')
        exit(1)
    qc = QuantumCircuit(2,name = 'G_prep')
    qc.ry(2*theta,0)
    qc.ch(0,1,ctrl_state = '0')          # Controlled Hadamard gate 
    return qc.to_gate()

def Block_encoding(n,dt,nu,shift_implementation):
    """
    n : number of spatial qubits
    dt : timestep
    nu : diffusion coefficent.
    
    Returns a circuit block encoding of the tridiagonal matrix with (1-2*nu*dt/dx**2) on the main diagonal
    and nu*dt/dx**2 on the two adjacent diagonals. The circuit has two registers qr1, qr2 both on n
    qubits and the matrix is encoded on the subspace |0...0> * \C^(2**n).
    """ 
    # Setting up the circuit 
    qr1 = QuantumRegister(n, name='Q1')
    qr2 = QuantumRegister(n, name='Q2')
    qc = QuantumCircuit(qr1,qr2, name = 'U_diff')
    
    # Preparing the needed gates 
    G = Diffusion_prep(n,dt,nu)
    S = shift_implementation(n)
    adder = ModularAdderGate(n)
    
    # Constructing the circuit 
    qc.append(G,qr1[:2])
    qc.append(S.inverse(),qr1[:])
    qc.append(adder,qr2[:]+qr1[:])
    
    for j in range(n):
        qc.swap(qr1[j],qr2[j])
    
    qc.append(adder.inverse(),qr2[:]+qr1[:])
    qc.append(S,qr1[:])
    qc.append(G.inverse(),qr1[:2])
    return qc    
    
# The following function extracts QSVT angle sequences from the file 'QSP_angles.txt'
# It contains the angle sequences for x^(5k) for 0<k<21

def extract_angle_seq(file = 'QSP_angles.txt'):
    """
    Returns a list of angle sequences 
    angle_seq[k] is the angle sequence of x**(5(k+1))
    """
    
    angle_seqs = []
    
    with open(file, 'r') as file:
        lines = file.readlines()[1:]

        for line in lines:
            seq = list(map(float, line.strip().split()))
            angle_seqs.append(np.array(seq))   
    return angle_seqs

# The following function implements QSVT on the block encoding provided by Block_encoding
# The full circuit is then simulated with the aer-simulator using Gaussian initial conditions. 
# A post-selection procedure picks out the successfull runs and arranges the results in a vector

def Diffusion_QSVT(deg,n,dt,nu,shots = 10**6,show_gate_count = False):
    """
    deg: number of time steps
    n: number of spatial qubits 
    dt: time step
    nu: diffusion coefficient
    shots: number of shots used in the aer-simulator
    show_gate_count: True or False according to whether gate counts should be printed
    
    The function implements QSVT for the function x**deg on Block_encoding(n,dt,nu)
    yielding a quantum circuit. The circuit is then measured in the computational basis. 
    
    The circuit is initalized with normalized Gaussian initial conditions over
    [0,d=4] with N = 2**n uniformly distributed grid points.  
    
    The circuit is simulated using the aer-simulator with shots = shots. 
    
    A post-selection procedure picks out the successfull runs and arranges the results in
    a vector z.

    Returns x,z the spatial grid values and the simulated y values in z. 
    """
    sim = AerSimulator()
    min_depth = 1_000_000_000
    min_idx = 0
    qc_comps = []
    depths = []
    qcs = []
    gates = []
    for idx,shift_implementation in enumerate(Shift_implementations): 
        # Setting up the circuit 
        qra = QuantumRegister(1)      # Ancilla register on 1 qubit used in QSVT 
        qr1 = QuantumRegister(n)      # qr1 and qr2 are the same as in Block_encoding 
        qr2 = QuantumRegister(n)
        
        cra = ClassicalRegister(1)
        cr1 = ClassicalRegister(n)
        cr2 = ClassicalRegister(n)
        
        qc = QuantumCircuit(qra,qr1,qr2,cra,cr1,cr2)
        
        # Preparing the initial conditions 
        N = 2**n 
        d = 4                                             # spatial domain [0,d]
        dx = 4/N 
        x = np.linspace(0,d,N,endpoint = False)
        y = np.exp(-20*(x-d/3)**2)                        # Gaussian initial conditions 
        y = y/np.linalg.norm(y)                           # normalized to be a unit vector
        qc.prepare_state(Statevector(y),qr2)
        U = Block_encoding(n,dt,nu,shift_implementation)                       # Block encoding circuit 
        Phi = extract_angle_seq()[int(deg/5)-1]           # Extracting the angle sequence  
        
        # Applying the QSVT circuit 
        qc.h(qra[0])
        s = 0
        for k in range(len(Phi)-1,-1,-1):
            if s == 0:
                qc.append(U,qr1[:]+qr2[:])
                s = 1
            else:
                qc.append(U.inverse(),qr1[:]+qr2[:])
                s = 0
            qc.mcx(qr1[:],qra[0],ctrl_state = n*'0')
            qc.rz(2*Phi[k],qra[0])
            qc.mcx(qr1[:],qra[0],ctrl_state = n*'0')
        qc.h(qra[0])
        
        # Measurements
        qc.measure(qra,cra)
        qc.measure(qr1,cr1)
        qc.measure(qr2,cr2)
        
        # Running the circuit     
        qc_comp = transpile(qc,sim)
        depth = qc_comp.depth()
        gate_1q = 0
        gate_2q = 0
        if depth <= min_depth:
            min_depth = depth
            min_idx = idx
        dict = qc_comp.count_ops()
        for key in dict:
            if key[0] == 'c':
                gate_2q += dict[key]
            elif key != 'measure':
                gate_1q += dict[key]

        qc_comps += [qc_comp]
        depths += [depth]
        qcs += [qc]
        gates += [(gate_1q, gate_2q)]
    #if show_gate_count:
    #    print("1 qubit gates:", gates[min_idx][0])
    #    print("2 qubit gates:", gates[min_idx][1])
    #    print("Total:", sum(gates[min_idx]))
    #    print('Circuit depth after transpiling:', depths[min_idx])
        
    # Postselection
    res = sim.run(qc_comps[min_idx],shots = shots).result()
    counts = res.get_counts(0)
    select = (n+1)*'0'
    total = 0                      # Tracks the number of successfull outcomes
    z = np.zeros(N)                # The results are encoded in z 
    for key in counts:
        L = key.split()
        if L[1]+L[2] == select:
            z[int(L[0],2)] = np.sqrt(counts[key]/shots)    # By construction all amplitudes are positive real numbers
            total += counts[key]                           # so this actually recovers them!
    success_rate = total/shots
    #print('Success rate =', success_rate)
    with open(f"{'_'.join([x.__name__ for x in Shift_implementations])}.dat", "a") as f:
        f.write(f"{deg}\t{n}\t{dt}\t{nu}\t{gates[min_idx][0]}\t{gates[min_idx][1]}\t{sum(gates[min_idx])}\t{depths[min_idx]}\t{success_rate}\n")
    return x,z

def Euler_cl(deg,n,dt,nu):
    """
    deg: number of time steps
    n: N=2**n is the number of spatial grid points
    dt: time step 
    nu: diffusion coefficient 
    
    Returns x,y,w where x are the spatial grid points, 
    y and z are the function values at x at time t=0 and t=deg*dt, respectively.
    """    
    N = 2**n
    d = 4                     # Domain [0,d]
    dx = d/N 
    x = np.linspace(0,d,N,endpoint = False)
    
    b = dt*nu/dx**2
    a = 1-2*dt*nu/(dx**2)
    B = b*np.diag(np.ones(N-1),-1)+a*np.diag(np.ones(N),0)+b*np.diag(np.ones(N-1),1)
    B[0][-1] = b 
    B[-1][0] = b
    
    y = np.exp(-20*(x-d/3)**2)
    y = y/np.linalg.norm(y)
    
    C = np.linalg.matrix_power(B,deg)
    w = np.matmul(C,y)
    return x,y,w 

def Compare_plots(deg = 10,n = 5,dt = 0.1,nu = 0.02,shots = 10**6):
    # Plots the initial distribution and the results of the classical and quantum simulations at t = deg*dt  
    x,y,w = Euler_cl(deg,n,dt,nu)  
    x,z = Diffusion_QSVT(deg,n,dt,nu,shots = shots, show_gate_count = True)
    T = deg*dt 
    plt.plot(x,y,x,w,x,z)
    plt.legend(['Classical T=0','Classical T='+str(T),'Quantum T='+str(T)])
#    plt.show()
Shift_implementations = []
for x in [[QFT_Shift_gate],[MCX_Shift_gate],[QFT_Shift_gate, MCX_Shift_gate]]:
    Shift_implementations = x 
    with open(f"{'_'.join([x.__name__ for x in Shift_implementations])}.dat", "w") as f:
        f.write("deg\tn\tdt\tnu\tgates_q1\tgates_q2\tgates_total\tcircuit_depth\tsucces_rate\n")

    for n in range(2,7):
        Compare_plots(deg = 20,n = n,dt = 0.05,nu = 0.02, shots = 10**6)

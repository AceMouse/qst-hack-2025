import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT, ModularAdderGate, RYGate, MCXGate
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

plot_style = {
    "displaytext": {
        # gate name : gate text to be displayed
        "block_adv_dg": "O_A^\dagger",
        "block_adv": "O_A",
        "c_right": "controlled right shift",
        "c_left": "controlled left shift",
    },

    "displaycolor": {

        "block_adv_dg": [ 
            "#18AA18", 
            "#000000" 
        ],

        "block_adv":  [ 
            "#ff1818", 
            "#000000" 
        ],
    },
}
"""import pyqsp
from pyqsp import angle_sequence, response
from pyqsp.poly import (polynomial_generators, PolyTaylorSeries)
deg = 20
# Specify definite-parity target function for QSP.
func = lambda x: x**deg
polydeg = 20 # Desired QSP protocol length.
max_scale = 0.9 # Maximum norm (<1) for rescaling.
true_func = lambda x: max_scale * func(x) # For error, include scale.


With PolyTaylorSeries class, compute Chebyshev interpolant to degree
'polydeg' (using twice as many Chebyshev nodes to prevent aliasing).

poly = PolyTaylorSeries().taylor_series(
    func=func,
    degree=polydeg,
    max_scale=max_scale,
    chebyshev_basis=True,
    cheb_samples=2*polydeg)

# Compute full phases (and reduced phases, parity) using symmetric QSP.
(phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
    poly,
    method='sym_qsp',
    chebyshev_basis=True)
"""
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

def Diffusion_block_encoding(n,dt,nu,shift_implementation):
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
    return qc.to_gate() 
    
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

def compute_angle_seq(file = 'QSP_angles.txt'):
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

def Advection_block_encoding(n, dt, c):
#    n = 4
#    dx = 0.1
#    dt = 0.001
#    c=1
    
    d = 4                               # The domain is fixed to be [0,4]
    dx = d/2**n                
    #qrl = QuantumRegister(n, name = "qrl")
    #cl = ClassicalRegister(3, name="Class")
    qra = QuantumRegister(3, name="Qa")
    qrn = QuantumRegister(n, name="Qn")
    beta = c*dt/(2*dx)
    if beta>1:
        print("Beta =c*dt/2dx needs to be <1")
        exit(1)
    theta0=2*np.acos(-beta)
    theta1 = 2*np.acos(beta)
    theta2=np.pi
    theta3=0
    
    phi0=0
    phi1=0
    phi2=0
    phi3=0
    
    #qc = QuantumCircuit(cl, qra, qrn, name = "block_adv")
    qc = QuantumCircuit(qra, qrn, name = "block_adv")
    qc.h(qra[1])
    qc.h(qra[2])
    
    """
    
    qc.x(qra[1])
    qc.x(qra[2])
    ry0 = RYGate(theta0).control(2)
    qc.append(ry0([1, 2, 0]))
    qc.x(qra[1])
    qc.x(qra[2])
    
    qc.x(gra[1])
    ry1 = RYGate(theta1).control(2)
    qc.append(ry0([1, 2, 0]))
    qc.x(qra[1])
    
    qc.x(gra[2])
    ry2 = RYGate(theta2).control(2)
    qc.append(ry2([1, 2, 0]))
    qc.x(qra[2])
    """
    
    ry0 = RYGate(theta0).control(2, ctrl_state = '00')
    ry1 = RYGate(theta1).control(2, ctrl_state = '10')
    ry2 = RYGate(theta2).control(2, ctrl_state = '01')
    
    qc.append(ry0, [qra[1], qra[2], qra[0]])
    qc.append(ry1, [qra[1], qra[2], qra[0]])
    qc.append(ry2, [qra[1], qra[2], qra[0]])
    
    
    
    
    #left shift
    L = QuantumCircuit(qrn, name = "left")
    
    for k in range(0, n-1):
        #L.append(MCXGate(num_ctrl_qubits = n-k-1), [qrl[k+1:n-1], qrl[k]])
        gate = MCXGate(num_ctrl_qubits = n-k-1)
        L.append(gate, list(range(k+1,n)) + [k])
        #L.append(gate, [qrl[k+1:n-1], qrl[k]])
        #L.append(gate, [qrl[k+1:n-1], qrl[k]])


    L.x(n-1)
    print(qc)
    print(L)
    
    R = QuantumCircuit(qrn, name = "right")
    for k in range(0, n-1):
        gate = MCXGate(num_ctrl_qubits = n-k-1, ctrl_state = (n-k-1)*
                       '0')
        R.append(gate, list(range(k+1, n)) + [k])
    
    print(R)
    
    
    LC = L.control().to_gate()
    RC = R.control().to_gate()
    
    qc.append(LC, list(range(2, n+3)))
    qc.append(RC, [1] +list(range(3, n+3)))
    
    qc.h(qra[1])
    qc.h(qra[2])
    #qc.barrier()
    #qc.measure(qra[0], cl[0])
    #qc.measure(qra[1], cl[1])
    #qc.measure(qra[2], cl[2])
#    qc.decompose(gates_to_decompose =["c_right", "c_left"]).draw(output="mpl",plot_barriers=False, fold=40, style=plot_style, scale=0.5).show()
#    L.draw(output="mpl",plot_barriers=False, fold=40, style=plot_style, scale=0.5).show()
#    R.draw(output="mpl",plot_barriers=False, fold=40, style=plot_style, scale=0.5).show()

#    input()
    
    return qc
# The following function implements QSVT on the block encoding provided by Block_encoding
# The full circuit is then simulated with the aer-simulator using Gaussian initial conditions. 
# A post-selection procedure picks out the successfull runs and arranges the results in a vector
def Advection_QSVT(deg,n,dt,c,shots = 10**6,show_gate_count = False):
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
    # Setting up the circuit 
    qra = QuantumRegister(1)      # Ancilla register on 1 qubit used in QSVT 
    qrab = QuantumRegister(3)      # This will store the ancillary qubits needed for the block encoding
    qrn = QuantumRegister(n)      #This will store the system qubits made in the block encoding
    
    cra = ClassicalRegister(1)
    crab = ClassicalRegister(3)
    crn = ClassicalRegister(n)
    
    qc = QuantumCircuit(qra, qrab, qrn, cra, crab, crn)
    
    # Preparing the initial conditions 
    N = 2**n 
    d = 4                                             # spatial domain [0,d]
    x = np.linspace(0,d,N,endpoint = False)
    y = np.exp(-20*(x-d/3)**2)                        # Gaussian initial conditions 
    y = y/np.linalg.norm(y)                           # normalized to be a unit vector
    qc.prepare_state(Statevector(y),qrn)
    U = Advection_block_encoding(n,dt,c)                       # Block encoding circuit 
    Phi = extract_angle_seq()[int(deg/5)-1]           # Extracting the angle sequence  
    
    # Applying the QSVT circuit 
    qc.h(qra[0])
    s = 0
    for k in range(len(Phi)-1,-1,-1):
        if s == 0:
            qc.append(U,qrab[:]+qrn[:])
            s = 1
        else:
            qc.append(U.inverse(),qrab[:]+qrn[:])
            s = 0
        qc.mcx(qrn[:],qra[0],ctrl_state = n*'0')
        qc.rz(2*Phi[k],qra[0])
        qc.mcx(qrn[:],qra[0],ctrl_state = n*'0')
    qc.h(qra[0])
    
    # Measurements
    qc.barrier()
    qc.measure(qra,cra)
    qc.measure(qrab,crab)
    qc.measure(qrn,crn)
    
    # Running the circuit     
#    qc.draw(output="mpl",plot_barriers=False, fold=40, scale=0.5, style=plot_style).show()
    qc_comp = transpile(qc,sim)
    gate_1q = 0
    gate_2q = 0
    dict = qc_comp.count_ops()
    for key in dict:
        if key[0] == 'c':
            gate_2q += dict[key]
        elif key != 'measure':
            gate_1q += dict[key]


    if show_gate_count:
        print("1 qubit gates:", gate_1q)
        print("2 qubit gates:", gate_2q)
        print("Total:", gate_1q + gate_2q )
        print('Circuit depth after transpiling:', qc_comp.depth())
        
    # Postselection
    res = sim.run(qc_comp,shots = shots).result()
    counts = res.get_counts(0)
    #plot_histogram(counts,legend=["counts"], color=['crimson','midnightblue'],
    #            title="New Histogram").show()
    select = (3+1)*'0'
    total = 0                      # Tracks the number of successfull outcomes
    z = np.zeros(N)                # The results are encoded in z 
    for key in counts:
        L = key.split()
        if L[1]+L[2] == select:
            z[int(L[0],2)] = np.sqrt(counts[key]/shots)    # By construction all amplitudes are positive real numbers
            total += counts[key]                           # so this actually recovers them!
    success_rate = total/shots
    print('Success rate =', success_rate)
    d=4
    x = np.linspace(0,d,N,endpoint = False)
    return x,z

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
        x = np.linspace(0,d,N,endpoint = False)
        y = np.exp(-20*(x-d/3)**2)                        # Gaussian initial conditions 
        y = y/np.linalg.norm(y)                           # normalized to be a unit vector
        qc.prepare_state(Statevector(y),qr2)
        U = Diffusion_block_encoding(n,dt,nu,shift_implementation)                       # Block encoding circuit 
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
        qc_comp = transpile(qc,sim,optimization_level=3)
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

def Diffusion_euler_cl(deg,n,dt,nu):
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

def Advection_euler_cl(deg,n,dt,c):
    """
    deg: number of time steps
    n: N=2**n is the number of spatial grid points
    dt: time step 
    c: speed coefficient 
    
    Returns x,y,w where x are the spatial grid points, 
    y and z are the function values at x at time t=0 and t=deg*dt, respectively.
    """    
    N = 2**n
    d = 4                     # Domain [0,d]
    dx = d/N 
    x = np.linspace(0,d,N,endpoint = False)
    
    b = c*dt/(2*dx)
    if (b>=1): 
        raise Exception(f"Advection error: beta over 1! beta = {b}")
    B = -b*np.diag(np.ones(N-1),-1)+b*np.diag(np.ones(N-1),1)
    B[0][-1] = b
    B[-1][0] = -b
    print(B)
    
    y = np.exp(-20*(x-d/3)**2)
    y = y/np.linalg.norm(y)
    
    C = np.linalg.matrix_power(B,deg)
    w = np.matmul(C,y)**2
    w = w/np.linalg.norm(w)
    return x,y,w+1  

def Diffusion_compare_plots(deg = 10,n = 5,dt = 0.1,nu = 0.02,shots = 10**6, plot=False):
    # Plots the initial distribution and the results of the classical and quantum simulations at t = deg*dt  
    x,y,w = Diffusion_euler_cl(deg,n,dt,nu)  
    x,z = Diffusion_QSVT(deg,n,dt,nu,shots = shots, show_gate_count = True)
    T = deg*dt 
    if (plot):
        plt.figure(6)
        plt.plot(x,y,x,w,x,z)
        plt.legend(['Classical T=0','Classical T='+str(T),'Quantum T='+str(T)])
        plt.show()

def Advection_compare_plots(deg = 10,n = 5,dt = 0.1,c = 3,shots = 10**6, plot=False):
    # Plots the initial distribution and the results of the classical and quantum simulations at t = deg*dt  
    #x,y,w = Advection_euler_cl(deg,n,dt,c)  
    x,y = Advection_QSVT(0,n,dt,c,shots = shots, show_gate_count = True)
    x,z = Advection_QSVT(deg,n,dt,c,shots = shots, show_gate_count = True)
    z = z/np.linalg.norm(z)
    T = deg*dt 
    if (plot):
        plt.figure(7)
        plt.plot(x,y,x,z)
        plt.legend(['Quantum T=0','Quantum T='+str(T)])
        plt.show()

Shift_implementations = [QFT_Shift_gate, MCX_Shift_gate]
#Diffusion_compare_plots(deg = 10, n = 6, dt = 0.05, nu = 0.02, shots = 10**6, plot=True)
Advection_compare_plots(deg = 1, n = 6, dt = 0.2, c = 0.2, shots = 10**6, plot=True)
"""for x in [[QFT_Shift_gate],[MCX_Shift_gate],[QFT_Shift_gate, MCX_Shift_gate]]:
    Shift_implementations = x 
    with open(f"{'_'.join([x.__name__ for x in Shift_implementations])}.dat", "w") as f:
        f.write("deg\tn\tdt\tnu\tgates_q1\tgates_q2\tgates_total\tcircuit_depth\tsucces_rate\n")
for n in range(2,10):
    for x in [[QFT_Shift_gate],[MCX_Shift_gate],[QFT_Shift_gate, MCX_Shift_gate]]:
        Shift_implementations = x 
        N = 2**n
        d = 4                     # Domain [0,d]
        dx = d/N
        nu = 0.02
        dt = ((dx**2)/(2*nu))/2
        print(f"dt: {dt}")
        Compare_plots(deg = 20,n = n,dt = dt,nu = nu, shots = 10**6)
"""

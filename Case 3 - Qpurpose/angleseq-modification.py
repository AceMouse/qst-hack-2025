import numpy as np
from pyqsp import angle_sequence
from pyqsp.poly import PolyTaylorSeries
from numpy.polynomial import polyutils as pu

"""
The python package pyqsp allows you to generate new angle sequences: https://github.com/ichuang/pyqsp

The sample code under "A guide within a guide" is the most efficient way to get started.
However, for use in the QSVT algorithm, 'phiset' as output by the sample code: 

(phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
    poly,
    method='sym_qsp',
    chebyshev_basis=True)

must be slightly modified. The function QSVT_format below implements the required
modification. 
"""	


# Specify definite-parity target function for QSP.

for deg in range(5,105,5):
    print(deg)
    func = lambda x: x**deg
    polydeg = deg # Desired QSP protocol length.
    max_scale = 0.9 # Maximum norm (<1) for rescaling.
    true_func = lambda x: max_scale * func(x) # For error, include scale.

    """
    With PolyTaylorSeries class, compute Chebyshev interpolant to degree
    'polydeg' (using twice as many Chebyshev nodes to prevent aliasing).
    """
    poly = PolyTaylorSeries().taylor_series(
        func=func,
        degree=polydeg,
        max_scale=max_scale,
        chebyshev_basis=True,
        cheb_samples=2*polydeg)
    for i in range(1-(deg%2),deg+1,2):
        poly.coef[i] = 0
    # Compute full phases (and reduced phases, parity) using symmetric QSP.
    (phiset1, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
        poly,
        method='sym_qsp',
        chebyshev_basis=True)
    #phiset2 = angle_sequence.QuantumSignalProcessingPhases(
    #    poly,
    #    method='tf')
    #phiset3 = np.array(angle_sequence.QuantumSignalProcessingPhases(poly,method='laurent'))

    def QSVT_format(phiset):
        n = len(phiset)-1
        Phi = np.zeros(n)
        Phi[1:n] = phiset[1:n]-np.pi/2
        Phi[0] = phiset[0]+phiset[-1]+((n-2)%4)*np.pi/2
        # If you use output from the MATLAB package qsppack you have to replace (n-2) by (n-1) in the above formula!
        return Phi
    with open("sym_qsp_angles.txt", "a") as f:    
        f.write(f"{QSVT_format(phiset1)%(2*np.pi)}\n")
    #print(QSVT_format(phiset2))
    #with open("laurent_angles.txt", "a") as f:    
    #    f.write(f"{QSVT_format(phiset3)}\n")
    

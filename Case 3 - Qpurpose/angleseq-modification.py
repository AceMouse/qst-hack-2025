import numpy as np

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

def QSVT_format(phiset):
	n = len(phiset)-1
	Phi = np.zeros(n)
	Phi[1:n] = phiset[1:n]-np.pi/2
	Phi[0] = phiset[0]+phiset[-1]+((n-2)%4)*np.pi/2
	# If you use output from the MATLAB package qsppack you have to replace (n-2) by (n-1) in the above formula!
	return Phi
    
    
    

import numpy as np

def E2lambda(E):
    '''
    Convert X-ray energy in keV to wavelength in Angstroms.
    '''
    return 12.39842/E

def lambda2E(l):
    '''
    Convert X-ray wavelength in Angstroms to energy in keV.
    '''
    return 12.39842/l

def theta2Q(Ei, theta):
    '''
    Convert the scattering angle in degrees to Q in 1/A.
    '''
    return 4*np.pi*np.sin(np.deg2rad(theta)/2)/E2lambda(Ei)

def Q2theta(Ei, Q):
    '''
    Convert Q in 1/A to the scattering angle in degrees.
    '''
    return 2*np.rad2deg(np.arcsin(E2lambda(Ei)*Q/4/np.pi))


def decorrelation_f(t, tau, beta, c, y0):
    '''
    Decorrelation function for XPCS analysis.
    '''
    return c * np.exp(-(t / tau) ** beta) + y0

import numpy as np

def E2lambda(E):
    '''
    Convert X-ray energy in keV to wavelength in Angstroms.
    
    Parameters
    ----------
        E: float
            X-ray energy in keV
    Returns
    -------
    float
        Wavelength in Angstroms
    '''
    return 12.39842/E

def lambda2E(l):
    '''
    Convert X-ray wavelength in Angstroms to energy in keV.

    Parameters
    ----------
        l: float
            Wavelength in Angstroms
    Returns
    -------
    float
        X-ray energy in keV
    '''
    return 12.39842/l

def theta2Q(Ei, theta):
    '''
    Convert the scattering angle in degrees to Q in 1/A.

    Parameters
    ----------
        Ei: float
            Energy of the beam in keV
        theta: float
            Scattering angle in degrees
    Returns
    -------
    float
        Q value in 1/A
    '''
    return 4*np.pi*np.sin(np.deg2rad(theta)/2)/E2lambda(Ei)

def Q2theta(Ei, Q):
    '''
    Convert Q in 1/A to the scattering angle in degrees.

    Parameters
    ----------
        Ei: float
            Energy of the beam in keV
        Q: float
            Q value in 1/A
    Returns
    -------   
    float
        Scattering angle in degrees
    '''
    return 2*np.rad2deg(np.arcsin(E2lambda(Ei)*Q/4/np.pi))


def decorrelation_f(t, tau, beta, c, y0):
    '''
    Decorrelation function for XPCS analysis.

    Parameters
    ----------
        t : array_like
            Time variable.
        tau : float
            Characteristic decay time.
        beta : float
            Stretching exponent.
        c : float
            Contrast or amplitude.
        y0 : float
            Baseline offset.
    Returns
    -------
    array_like
        Decorrelation function evaluated at t.
    '''
    return c * np.exp(-(t / tau) ** beta) + y0

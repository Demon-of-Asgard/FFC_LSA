import sys 
import numpy as np
import scipy.integrate as sint
import matplotlib.pyplot as plt

from src import nma_3d as nma

vzmin, vzmax = -1, 1 
phmin, phmax = -np.pi, np.pi 
nvz, nph = 32, 8
dvz, dph = (vzmax - vzmin) / nvz, (phmax - phmin) / nph
vzs = vzmin + (0.5 + np.arange(nvz)) * dvz
phs = phmin + (0.5 + np.arange(nph)) * dph
fluxangle_deg = 30

def f(vz, ph=0, sig=0.6):
    return np.exp(-(vz-1)**2/(2*sig**2))

def ELN(vz, phi, signu=0.6, siganu=0.5, alpha=0.9, 
        N=0.7513431855316718, bN=0.6266173746426144, 
        relative_fluxangle=fluxangle_deg*(np.pi/180.0)):

    vx = np.sqrt(1.0 - vz**2) * np.cos(phi)
    vy = np.sqrt(1.0 - vz**2) * np.sin(phi)

    cr = np.cos(relative_fluxangle)
    sr = np.sin(relative_fluxangle)

    vzr = -sr * vx + cr * vz

    return f(vz=vz, sig=signu)/N - alpha * f(vz=vzr, sig=siganu) / bN

def main():
    eln_params = {
        'ELN' : ELN,
        'vz_range' : (vzmin, vzmax),
        'phi_range' : (phmin, phmax),
        'nvz': nvz,
        'nphi':nph,
    }

    kx = np.fft.fftshift(np.fft.fftfreq(n=100, d=1.0)) 
    ky = np.fft.fftshift(np.fft.fftfreq(n=100, d=1.0)) 
    kz = np.fft.fftshift(np.fft.fftfreq(n=100, d=1.0)) 

    lsa = nma.NMA_3D(ELN_params=eln_params)

    store_to = lsa.run(
        kxs=kx, 
        kys=ky, #np.array([0,]), 
        kzs=kz,
        store_to=f"lsa_{fluxangle_deg:.0f}.dat",
    )

    print (f"Results dtored at {store_to}")


if __name__ == "__main__":
    main()
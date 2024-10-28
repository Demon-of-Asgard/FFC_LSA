import sys
import numpy as np
import scipy.integrate as sint
import matplotlib.pyplot as plt
from typing import (Union, List, Tuple)

# Import nma library
from src import nma_3d as nma

# specify angular grids and
# angular distribution parameters
vzmin, vzmax = -1, 1
phmin, phmax = -np.pi, np.pi
nvz, nph = 32, 8
dvz, dph = (vzmax - vzmin) / nvz, (phmax - phmin) / nph
vzs = vzmin + (0.5 + np.arange(nvz)) * dvz
phs = phmin + (0.5 + np.arange(nph)) * dph

fluxangle = 30 * np.pi / 180.0


# Angular distribution
def f(vz, ph=0, sig=0.6):
    return np.exp(-(vz-1)**2/(2*sig**2))


# ELN
def ELN(vz: Union[float, np.ndarray[float]], phi: Union[float, np.ndarray[float]],
        signu: float = 0.6, siganu: float = 0.5, alpha: float = 0.9,
        N: float = 0.7513431855316718, bN: float = 0.6266173746426144,
        relative_fluxangle=fluxangle) -> Union[float, np.ndarray[float]]:

    Fn_norm = 0.5227104262667555
    Fan_norm = 0.6011662867063625

    arg1 = np.sin(relative_fluxangle) / \
        ((Fn_norm/Fan_norm/alpha) - np.cos(relative_fluxangle))
    coord_rotangle = np.arctan(arg1)

    ang_nu = coord_rotangle
    ang_anu = coord_rotangle + relative_fluxangle

    vx = np.sqrt(1.0 - vz**2) * np.cos(phi)
    vy = np.sqrt(1.0 - vz**2) * np.sin(phi)

    cr_n = np.cos(ang_nu)
    sr_n = np.sin(ang_nu)

    cr_an = np.cos(ang_anu)
    sr_an = np.sin(ang_anu)

    vzr_n = -sr_n * vx + cr_n * vz
    vzr_an = -sr_an * vx + cr_an * vz

    return f(vz=vzr_n, sig=signu)/N - alpha * f(vz=vzr_an, sig=siganu) / bN


# ELN AXIAPPROXIMATION
def ELN_axi(vz: Union[float, np.ndarray[float]], phi: Union[float, np.ndarray[float]],
            signu: float = 0.6, siganu: float = 0.5, alpha: float = 0.9,
            N: float = 0.7513431855316718, bN: float = 0.6266173746426144,
            relative_fluxangle=fluxangle) -> Union[float, np.ndarray[float]]:

    Fn_norm = 0.5227104262667555
    Fan_norm = 0.6011662867063625

    arg1 = np.sin(relative_fluxangle) / \
        ((Fn_norm/Fan_norm/alpha) - np.cos(relative_fluxangle))
    coord_rotangle = np.arctan(arg1)

    ang_nu = coord_rotangle
    ang_anu = coord_rotangle + relative_fluxangle

    fnu = np.zeros(nph)
    fanu = np.zeros(nph)
    for phid, phi in enumerate(phs):
        vx = np.sqrt(1.0 - vz**2) * np.cos(phi)
        vy = np.sqrt(1.0 - vz**2) * np.sin(phi)

        cr_n = np.cos(ang_nu)
        sr_n = np.sin(ang_nu)

        cr_an = np.cos(ang_anu)
        sr_an = np.sin(ang_anu)

        vzr_n = -sr_n * vx + cr_n * vz
        vzr_an = -sr_an * vx + cr_an * vz
        fnu[phid] = f(vz=vzr_n, sig=signu)/N
        fanu[phid] = f(vz=vzr_an, sig=siganu) / bN

    return (fnu - alpha * fanu).sum() / nph


def main():
    eln_params = {
        'ELN': ELN_axi,
        'vz_range': (vzmin, vzmax),
        'phi_range': (phmin, phmax),
        'nvz': nvz,
        'nphi': nph,
    }

    kx = np.fft.fftshift(np.fft.fftfreq(n=100, d=1.0))
    ky = np.fft.fftshift(np.fft.fftfreq(n=100, d=1.0))
    kz = np.fft.fftshift(np.fft.fftfreq(n=100, d=1.0))

    lsa = nma.NMA_3D(ELN_params=eln_params)

    store_to = f"lsa_rot_axi_{fluxangle*(180/np.pi):.0f}.dat"

    lsa.run(
        kxs=kx,
        kys=ky,
        kzs=kz,
        store_to=store_to,
    )

    print(f"Results stored at {store_to}")


if __name__ == "__main__":
    main()

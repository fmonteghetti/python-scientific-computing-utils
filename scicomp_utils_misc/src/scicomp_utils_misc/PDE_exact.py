#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functions giving exact solutions for basic partial
differential equations (PDE). It is mostly useful for validation purposes.
"""

import numpy as np
from scicomp_utils_misc import scipy_utils


class Laplace_spectrum:
    """Spectrum of Laplace operator u -> -div(grad(u))."""

    def interval(Lx, N, bc):
        """
        Spectrum of Laplace operator on (0,Lx).

        Parameters
        ----------
        Lx : float
        N : int
            Number of eigenvalues
        bc : string
            'dirichlet', 'neumann', 'periodic'
        """
        eigval = []
        if bc == "dirichlet":
            eigval = [(nx * np.pi / Lx) ** 2 for nx in range(1, N + 1)]
        elif bc == "neumann":
            eigval = [(nx * np.pi / Lx) ** 2 for nx in range(0, N)]
        elif bc == "periodic":
            eigval = [(2 * nx * np.pi / Lx) ** 2 for nx in range(0, N)]
        return np.array(eigval)

    def rectangle(Lx, Ly, N, bc):
        """Spectrum of Laplace operator on (0,Lx)x(0,Ly)."""
        eigval = []
        if bc == "dirichlet":
            eigval = [
                (nx * np.pi / Lx) ** 2 + (ny * np.pi / Ly) ** 2
                for nx in range(1, N + 1)
                for ny in range(1, N + 1)
            ]
        elif bc == "neumann":
            eigval = [
                (nx * np.pi / Lx) ** 2 + (ny * np.pi / Ly) ** 2
                for nx in range(0, N)
                for ny in range(0, N)
                if (nx + ny) >= 1
            ]
            eigval = np.append(eigval, np.array([0]))
        return np.sort(eigval)[0:N]

    def cube(Lx, Ly, Lz, N, bc):
        """Spectrum of Laplace operator on (0,Lx)x(0,Ly)x(0,Lz)."""
        eigval = []
        if bc == "dirichlet":
            eigval = [
                (nx * np.pi / Lx) ** 2
                + (ny * np.pi / Ly) ** 2
                + (nz * np.pi / Lz) ** 2
                for nx in range(1, N + 1)
                for ny in range(1, N + 1)
                for nz in range(1, N + 1)
            ]
        elif bc == "neumann":
            eigval = [
                (nx * np.pi / Lx) ** 2
                + (ny * np.pi / Ly) ** 2
                + (nz * np.pi / Lz) ** 2
                for nx in range(N)
                for ny in range(N)
                for nz in range(N)
                if nx + ny + nz >= 1
            ]
            eigval = np.append(eigval, np.array([0]))
        return np.sort(eigval)[0:N]

    def disc(
        R, bc, eigval_min=0, eigval_max=10, eigval_N=10, eigval_maxorder=2
    ):
        """
        Spectrum of Laplace operator on 2D disc of radius R.

        Parameters
        ----------
        R : float
        bc : string
            'dirichlet' or 'neumann'
        eigval_min : float, optional
            Minimum eigenvalue. The default is 0.
        eigval_max : float, optional
            Maximum eigenvalue. The default is 10.
        eigval_N : int, optional
            Number of eigenvalues. The default is 10.
        eigval_maxorder : TYPE, optional
            Maximum eigenfunction order. The default is 1.
        """
        import scipy.special

        z = np.array([])
        if bc == "dirichlet":
            # Bessel function of first-kind
            f = lambda n, x: scipy.special.jv(n, x)
        elif bc == "neumann":
            # Derivative of Bessel function of first-kind
            f = lambda n, x: scipy.special.jvp(n, x)
        else:
            return []
        for n in range(eigval_maxorder):  # Bessel function order
            xmin, xmax = R * np.sqrt(eigval_min), R * np.sqrt(eigval_max)
            dx = (xmax - xmin) / eigval_N
            zeros = scipy_utils.find_zeros(lambda x: f(n, x), xmin, xmax, dx)
            z = np.append(z, zeros)
        z = np.sort(z)
        if bc == "dirichlet":  # remove 0
            z = z[np.abs(z) > 1e-10]
        return (z / R) ** 2
        pass

    def ball(
        R, bc, eigval_min=0, eigval_max=10, eigval_N=10, eigval_maxorder=2
    ):
        """
        Spectrum of Laplace operator on 3D ball of radius R.

        Parameters
        ----------
        R : float
        bc : string
            'dirichlet' or 'neumann'
        eigval_min : float, optional
            Minimum eigenvalue. The default is 0.
        eigval_max : float, optional
            Maximum eigenvalue. The default is 10.
        eigval_N : int, optional
            Number of eigenvalues. The default is 10.
        eigval_maxorder : TYPE, optional
            Maximum eigenfunction order. The default is 1.
        """
        import scipy.special

        z = np.array([])
        if bc == "dirichlet":
            # Spherical Bessel function of first-kind
            f = lambda n, x: scipy.special.spherical_jn(n, x)
        elif bc == "neumann":
            # Derivative of spherical Bessel function of first-kind
            f = lambda n, x: scipy.special.spherical_jn(n, x, derivative=True)
        else:
            return []
        for n in range(eigval_maxorder + 1):  # Bessel function order
            xmin, xmax = R * np.sqrt(eigval_min), R * np.sqrt(eigval_max)
            dx = (xmax - xmin) / eigval_N
            zeros = scipy_utils.find_zeros(lambda x: f(n, x), xmin, xmax, dx)
            z = np.append(z, zeros)
        z = np.sort(z)
        if bc == "dirichlet":  # remove 0
            z = z[np.abs(z) > 1e-10]
        return (z / R) ** 2


class Maxwell_spectrum:
    """Spectrum of Maxwell operator: (E,H) -> (curl(H),-curl(E)).
    Three cases are covered:
    (i) 3D:
        - E and H are vector fields with null divergence.
        - PEC condition: E has a null tangential trace.
    (ii) 2D transverse electric:
        - E is a bi-dimensional vector field with a null
    bi-dimensional divergence, H is a scalar field.
        - PEC condition: E has a null tangential trace.
    (iii) 2D transverse magnetic:
        - E is a scalar field, H is a bi-dimensional vector field with a null
    bi-dimensional divergence.
        - PEC condition: E has a null trace, H has a null normal trace.
    """

    def cube(Lx, Ly, Lz, N, bc):
        """Spectrum of Maxwell operator on (0,Lx)x(0,Ly)x(0,Lz)."""
        eigval = []
        if bc == "pec":
            # Laplace modes
            eigval_LN_1D = Laplace_spectrum.interval(Lz, N, "neumann")
            eigval_LD_1D = Laplace_spectrum.interval(Lz, N, "dirichlet")
            # only non-zero neumann eigenvalues
            eigval_LN_2D = Laplace_spectrum.rectangle(Lx, Ly, N, "neumann")[1:]
            eigval_LD_2D = Laplace_spectrum.rectangle(Lx, Ly, N, "dirichlet")
            # Transverse elecric modes
            eigval_TE = [a + b for a in eigval_LN_2D for b in eigval_LD_1D]
            # Transverse magnetic modes
            eigval_TM = [a + b for a in eigval_LD_2D for b in eigval_LN_1D]
            eigval = np.append(eigval_TE, eigval_TM)
            eigval = 1j * np.sort(np.sqrt(eigval))
        return eigval

    def ball(
        R, bc, eigval_min=0, eigval_max=10, eigval_N=10, eigval_maxorder=4
    ):
        """Spectrum of Maxwell operator on ball of radius R."""
        import scipy.special

        eigval = np.array([])
        if bc == "pec":
            # Ricatti-Bessel function of the first kind
            f = lambda n, x: x * scipy.special.spherical_jn(n, x)
            # Derivative of Ricatti-Bessel function of the first kind
            g = lambda n, x: scipy.special.spherical_jn(
                n, x
            ) - x * scipy.special.spherical_jn(n, x, derivative=True)
            for n in range(eigval_maxorder + 1):  # Bessel function order
                xmin, xmax = R * eigval_min, R * eigval_max
                dx = (xmax - xmin) / eigval_N
                # Transverse electric modes
                eigval_TE = scipy_utils.find_zeros(
                    lambda x: f(n, x), xmin, xmax, dx
                )
                # Transverse magnetic modes
                eigval_TM = scipy_utils.find_zeros(
                    lambda x: g(n, x), xmin, xmax, dx
                )
                eigval = np.append(eigval, eigval_TM)
                eigval = np.append(eigval, eigval_TE)
            eigval = eigval[np.abs(eigval) > 1e-10] / R  # Remove 0
            eigval = 1j * np.sort(eigval)
        return eigval

    def rectangle_TE(Lx, Ly, N, bc):
        """Spectrum of transverse electric Maxwell operator on (0,Lx)x(0,Ly)."""
        eigval = []
        if bc == "pec":
            # Non-zero Neumann Laplacian eigenvalues
            eigval = Laplace_spectrum.rectangle(Lx, Ly, N, "neumann")[1:]
            eigval = 1j * np.sort(np.sqrt(eigval))
        return eigval

    def rectangle_TM(Lx, Ly, N, bc):
        """Spectrum of transverse magnetic Maxwell operator on (0,Lx)x(0,Ly)."""
        eigval = []
        if bc == "pec":
            # Dirichlet Laplacian eigenvalues
            eigval = Laplace_spectrum.rectangle(Lx, Ly, N, "dirichlet")
            eigval = 1j * np.sort(np.sqrt(eigval))
        return eigval

    def disc_TE(
        R, bc, eigval_min=0, eigval_max=10, eigval_N=10, eigval_maxorder=2
    ):
        """
        Spectrum of transverse electric Maxwel operator on 2D disc of radius R.

        Parameters
        ----------
        R : float
        bc : string
            'dirichlet' or 'neumann'
        eigval_min : float, optional
            Minimum eigenvalue. The default is 0.
        eigval_max : float, optional
            Maximum eigenvalue. The default is 10.
        eigval_N : int, optional
            Number of eigenvalues. The default is 10.
        eigval_maxorder : TYPE, optional
            Maximum eigenfunction order. The default is 1.
        """
        eigval = []
        if bc == "pec":
            # Non-zero Neumann Laplacian eigenvalues
            eigval = Laplace_spectrum.disc(
                R,
                "neumann",
                eigval_min=eigval_min,
                eigval_max=eigval_max,
                eigval_N=eigval_N,
                eigval_maxorder=eigval_maxorder,
            )[1:]
            eigval = 1j * np.sort(np.sqrt(eigval))
        return eigval

    def disc_TM(
        R, bc, eigval_min=0, eigval_max=10, eigval_N=10, eigval_maxorder=2
    ):
        """
        Spectrum of transverse magnetic Maxwel operator on 2D disc of radius R.

        Parameters
        ----------
        R : float
        bc : string
            'dirichlet' or 'neumann'
        eigval_min : float, optional
            Minimum eigenvalue. The default is 0.
        eigval_max : float, optional
            Maximum eigenvalue. The default is 10.
        eigval_N : int, optional
            Number of eigenvalues. The default is 10.
        eigval_maxorder : TYPE, optional
            Maximum eigenfunction order. The default is 1.
        """
        eigval = []
        if bc == "pec":
            # Neumann Dirichlet eigenvalues
            eigval = Laplace_spectrum.disc(
                R,
                "dirichlet",
                eigval_min=eigval_min,
                eigval_max=eigval_max,
                eigval_N=eigval_N,
                eigval_maxorder=eigval_maxorder,
            )
            eigval = 1j * np.sort(np.sqrt(eigval))
        return eigval

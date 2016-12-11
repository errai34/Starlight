#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
cimport numpy as np
import numpy as np
from cython.parallel import prange
from cpython cimport bool
cimport cython
from libc.math cimport sqrt, M_PI, exp, pow, erf, log, log10
from libc.stdlib cimport abort, malloc, free

cdef double gauss_prob(double x, double mu, double sig) nogil:
    return exp(- 0.5 * pow((x - mu)/sig, 2.)) / (sqrt(2.*M_PI) * sig)

cdef double gauss_prob_grad(double x, double mu, double sig) nogil:
    return - gauss_prob(x, mu, sig) * (x - mu)/sig**2

cdef double gauss_lnprob(double x, double mu, double sig) nogil:
    return 0.5 * pow((x - mu)/sig, 2) + 0.5 * log(2*M_PI) + log(sig)

cdef double gauss_lnprob_grad_x(double x, double mu, double sig) nogil:
    return (x - mu) / (sig * sig)

cdef double gaussdistmag_lnprob(
    double dist, double absmag, double obsmag, double obsmagg_err) nogil:
    return 0.5 * pow((absmag + 5.*log10(dist) + 10. - obsmag)/obsmagg_err, 2.) + 0.5 * log(2*M_PI) + log(obsmagg_err)

cdef double gaussdistmag_lnprob_grad_dist(double dist, double absmag, double obsmag, double obsmagg_err) nogil:
    return 5. * (absmag + 5*log10(dist) + 10. - obsmag) / (dist * obsmagg_err * obsmagg_err * log(10.))

cdef double gaussdistmag_lnprob_grad_absmag(double dist, double absmag, double obsmag, double obsmagg_err) nogil:
    return (absmag + 5*log10(dist) + 10. - obsmag) / (obsmagg_err * obsmagg_err)

cdef double gaussdistvarpi_lnprob(double dist, double varpi, double varpi_err) nogil:
    return 0.5 * pow((1/dist - varpi)/varpi_err, 2.) + 0.5 * log(2*M_PI) + log(varpi_err)

cdef double gaussdistvarpi_lnprob_grad_dist(double dist, double varpi, double varpi_err) nogil:
    return - (1./dist - varpi) / (varpi_err*varpi_err*dist*dist)


def lnprob_nomarg(
    long nobj,
    long nbins,
    long ncols,
    double[:] varpi,  # nobj
    double[:] varpi_err,  # nobj
    double[:] obsmags,  # nobj
    double[:] obsmags_err,  # nobj
    double[:, :] obscolors,  # nobj, ncols
    double[:, :] obscolors_err,  # nobj, ncols
    double[:] absmags,  # nobj
    double[:] distances,  # nobj
    double[:, :] colors,  # nobj, ncols
    double[:] binamps,  # nbins
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :] binsigs  # nbins, ncols + 1
    ):
    cdef double valtot = 0, probbins, probker
    cdef long b, j, o
    for o in prange(nobj, nogil=True):
        # distance and absmags contribution
        valtot +=\
            gaussdistvarpi_lnprob(distances[o], varpi[o], varpi_err[o]) +\
            gaussdistmag_lnprob(distances[o], absmags[o], obsmags[o], obsmags_err[o])
        # color contributions
        for j in range(ncols):
            valtot += gauss_lnprob(colors[o, j], obscolors[o, j], obscolors_err[o, j])
        # density contributions
        probbins = 0
        for b in range(nbins):
            probker = gauss_prob(absmags[o], binmus[b, 0], binsigs[b, 0])
            for j in range(ncols):
                probker = probker * gauss_prob(colors[o, j], binmus[b, j+1], binsigs[b, j+1])
            probbins = probbins + binamps[b] * probker / nbins
        valtot += - log(probbins)
    return valtot


def lnprob_gradients_nomarg(
    double[:] absmags_grad,  # nobj
    double[:] distances_grad,  # nobj
    double[:, :] colors_grad,  # nobj, ncols
    double[:] binamps_grad,  # nbins
    long nobj,
    long nbins,
    long ncols,
    double[:] varpi,  # nobj
    double[:] varpi_err,  # nobj
    double[:] obsmags,  # nobj
    double[:] obsmags_err,  # nobj
    double[:, :] obscolors,  # nobj, ncols
    double[:, :] obscolors_err,  # nobj, ncols
    double[:] absmags,  # nobj
    double[:] distances,  # nobj
    double[:, :] colors,  # nobj, ncols
    double[:] binamps,  # nbins
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :] binsigs  # nbins, ncols + 1
    ):
    cdef long i, b, j, o
    cdef double probbins, probker
    for b in range(nbins):
        binamps_grad[b] = 0
    for o in prange(nobj, nogil=True):
        # distance contribution
        distances_grad[o] = gaussdistvarpi_lnprob_grad_dist(distances[o], varpi[o], varpi_err[o])
        # absmags contribution
        distances_grad[o] += gaussdistmag_lnprob_grad_dist(distances[o], absmags[o], obsmags[o], obsmags_err[o])
        absmags_grad[o] = gaussdistmag_lnprob_grad_absmag(distances[o], absmags[o], obsmags[o], obsmags_err[o])
        # color contributions
        for j in range(ncols):
            colors_grad[o, j] = gauss_lnprob_grad_x(colors[o, j], obscolors[o, j], obscolors_err[o, j])
        # density contributions
        probbins = 0
        for b in range(nbins):
            probker = gauss_prob(absmags[o], binmus[b, 0], binsigs[b, 0])
            for j in range(ncols):
                probker = probker * gauss_prob(colors[o, j], binmus[b, j+1], binsigs[b, j+1])
            probbins = probbins + binamps[b] * probker / nbins
        for b in range(nbins):
            probker = gauss_prob_grad(absmags[o], binmus[b, 0], binsigs[b, 0])
            for j in range(ncols):
                probker = probker * gauss_prob(colors[o, j], binmus[b, j+1], binsigs[b, j+1])
            absmags_grad[o] += - binamps[b] * probker / nbins / probbins
        for j in range(ncols):
            for b in range(nbins):
                probker = gauss_prob(absmags[o], binmus[b, 0], binsigs[b, 0])
                for i in range(ncols):
                    if i == j:
                        probker = probker * gauss_prob_grad(colors[o, i], binmus[b, i+1], binsigs[b, i+1])
                    else:
                        probker = probker * gauss_prob(colors[o, i], binmus[b, i+1], binsigs[b, i+1])
                colors_grad[o, j] += - binamps[b] * probker / nbins / probbins
        for b in range(nbins):
            probker = gauss_prob(absmags[o], binmus[b, 0], binsigs[b, 0])
            for j in range(ncols):
                probker = probker * gauss_prob(colors[o, j], binmus[b, j+1], binsigs[b, j+1])
            with gil:
                binamps_grad[b] += - probker / nbins / probbins


def lnprob_marg(
    long nobj, long nbins, long ncols,
    double[:] varpi,  # nobj
    double[:] varpi_err,  # nobj
    double[:] obsmags,  # nobj
    double[:] obsmags_err,  # nobj
    double[:, :] obscolors,  # nobj, ncols
    double[:, :] obscolors_err,  # nobj, ncols
    long[:] bins,  # nobj
    double[:] distances,  # nobj
    double[:] binamps,  # nbins
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :] binsigs  # nbins, ncols + 1
    ):
    cdef double valtot = 0, sig
    cdef long b, j, o
    for o in prange(nobj, nogil=True):
        b = bins[o]
        valtot += - log(binamps[b] / nbins)
        valtot += gaussdistvarpi_lnprob(distances[o], varpi[o], varpi_err[o])
        sig = sqrt(pow(obsmags_err[o], 2) + pow(binsigs[b, 0], 2))
        valtot += gauss_lnprob(5.*log10(distances[o]) + 10.,
                               obsmags[o] - binmus[b, 0], sig)
        for j in range(ncols):
            sig = sqrt(pow(obscolors_err[o, j], 2) + pow(binsigs[b, j+1], 2))
            valtot += gauss_lnprob(obscolors[o, j], binmus[b, j+1], sig)
    return valtot


def prob_grid_marg(
    double[:, :] probgrid,  # nobj, nbins
    long nobj, long nbins, long ncols,
    double[:] varpi,  # nobj
    double[:] varpi_err,  # nobj
    double[:] obsmags,  # nobj
    double[:] obsmags_err,  # nobj
    double[:, :] obscolors,  # nobj, ncols
    double[:, :] obscolors_err,  # nobj, ncols
    double[:] distances,  # nobj
    double[:] binamps,  # nbins
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :] binsigs  # nbins, ncols + 1
    ):
    cdef double valtot = 0, sig
    cdef long b, j, o
    for o in prange(nobj, nogil=True):
        for b in range(nbins):
            probgrid[o, b] = gauss_prob(1/distances[o], varpi[o], varpi_err[o])
            sig = sqrt(pow(obsmags_err[o], 2) + pow(binsigs[b, 0], 2))
            probgrid[o, b] *= binamps[b] / nbins * gauss_prob(5*log10(distances[o]) + 10, obsmags[o] - binmus[b, 0], sig)
            for j in range(ncols):
                sig = sqrt(pow(obscolors_err[o, j], 2) + pow(binsigs[b, j+1], 2))
                probgrid[o, b] *= gauss_prob(obscolors[o, j], binmus[b, j+1], sig)


def lnprob_distgradient_marg(
    double[:] distances_grad,  # nobj
    long nobj, long nbins, long ncols,
    double[:] varpi,  # nobj
    double[:] varpi_err,  # nobj
    double[:] obsmags,  # nobj
    double[:] obsmags_err,  # nobj
    double[:, :] obscolors,  # nobj, ncols
    double[:, :] obscolors_err,  # nobj, ncols
    long[:] bins,  # nobj
    double[:] distances,  # nobj
    double[:] binamps,  # nbins
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :] binsigs  # nbins, ncols + 1
    ):
    cdef double sig
    cdef long b, o
    for o in prange(nobj, nogil=True):
        distances_grad[o] = gaussdistvarpi_lnprob_grad_dist(distances[o], varpi[o], varpi_err[o])
        b = bins[o]
        sig = sqrt(pow(obsmags_err[o], 2) + pow(binsigs[b, 0], 2))
        distances_grad[o] +=\
            (5*log10(distances[o]) + 10 - obsmags[o] + binmus[b, 0])\
            * 5 / (sig*sig*distances[o]*log(10.))

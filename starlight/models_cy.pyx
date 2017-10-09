#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
cimport numpy as np
import numpy as np
from cython.parallel import prange
from cpython cimport bool
cimport cython
from libc.math cimport sqrt, M_PI, exp, pow, erf, log, log10
from libc.stdlib cimport abort, malloc, free
cimport scipy.linalg.cython_lapack as cython_lapack

cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)

cdef extern from "gsl/gsl_randist.h":
    double gaussian "gsl_ran_gaussian"(gsl_rng * r,double) nogil
    double uniform "gsl_ran_flat"(const gsl_rng * r, double, double) nogil

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
            probbins = probbins + binamps[b] * probker
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
            probbins = probbins + binamps[b] * probker
        for b in range(nbins):
            probker = gauss_prob_grad(absmags[o], binmus[b, 0], binsigs[b, 0])
            for j in range(ncols):
                probker = probker * gauss_prob(colors[o, j], binmus[b, j+1], binsigs[b, j+1])
            absmags_grad[o] += - binamps[b] * probker / probbins
        for j in range(ncols):
            for b in range(nbins):
                probker = gauss_prob(absmags[o], binmus[b, 0], binsigs[b, 0])
                for i in range(ncols):
                    if i == j:
                        probker = probker * gauss_prob_grad(colors[o, i], binmus[b, i+1], binsigs[b, i+1])
                    else:
                        probker = probker * gauss_prob(colors[o, i], binmus[b, i+1], binsigs[b, i+1])
                colors_grad[o, j] += - binamps[b] * probker / probbins
        for b in range(nbins):
            probker = gauss_prob(absmags[o], binmus[b, 0], binsigs[b, 0])
            for j in range(ncols):
                probker = probker * gauss_prob(colors[o, j], binmus[b, j+1], binsigs[b, j+1])
            with gil:
                binamps_grad[b] += - probker / probbins


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
        valtot += - log(binamps[b])
        valtot += gaussdistvarpi_lnprob(distances[o], varpi[o], varpi_err[o])
        sig = sqrt(pow(obsmags_err[o], 2) + pow(binsigs[b, 0], 2))
        valtot += gauss_lnprob(5.*log10(distances[o]) + 10.,
                               obsmags[o] - binmus[b, 0], sig)
        for j in range(ncols):
            sig = sqrt(pow(obscolors_err[o, j], 2) + pow(binsigs[b, j+1], 2))
            valtot += gauss_lnprob(obscolors[o, j], binmus[b, j+1], sig)
    return valtot


def prob_distgrid_marg(
    double[:, :] probgrid,  # nobj, nbins
    long nbinsdist, double[:] distances_grid,  # nobj
    long nobj, long nbins, long ncols,
    double[:] varpi,  # nobj
    double[:] varpi_err,  # nobj
    double[:] obsmags,  # nobj
    double[:] obsmags_err,  # nobj
    double[:, :] obscolors,  # nobj, ncols
    double[:, :] obscolors_err,  # nobj, ncols
    double[:] binamps,  # nbins
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :] binsigs  # nbins, ncols + 1
    ):
    cdef double valtot = 0, sig, probgridterm
    cdef long b, j, o
    for o in prange(nobj, nogil=True):
        for b in range(nbins):
            probgridterm = binamps[b]
            for j in range(ncols):
                sig = sqrt(pow(obscolors_err[o, j], 2) + pow(binsigs[b, j+1], 2))
                probgridterm *= gauss_prob(obscolors[o, j], binmus[b, j+1], sig)
            sig = sqrt(pow(obsmags_err[o], 2) + pow(binsigs[b, 0], 2))
            for j in range(nbinsdist):
                probgrid[o, j] += probgridterm * gauss_prob(1/distances_grid[j], varpi[o], varpi_err[o]) * gauss_prob(5*log10(distances_grid[j]) + 10, obsmags[o] - binmus[b, 0], sig)


def prob_distgrids_marg(
    double[:, :] probgrid,  # nobj, nbins
    long nbinsdist, double[:, :] distance_grids,  # nobj, nbinsdist
    long nobj, long nbins, long ncols,
    double[:] varpi,  # nobj
    double[:] varpi_err,  # nobj
    double[:] obsmags,  # nobj
    double[:] obsmags_err,  # nobj
    double[:, :] obscolors,  # nobj, ncols
    double[:, :] obscolors_err,  # nobj, ncols
    double[:] binamps,  # nbins
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :] binsigs  # nbins, ncols + 1
    ):
    cdef double valtot = 0, sig, probgridterm
    cdef long b, j, o
    for o in prange(nobj, nogil=True):
        for b in range(nbins):
            probgridterm = binamps[b]
            for j in range(ncols):
                sig = sqrt(pow(obscolors_err[o, j], 2) + pow(binsigs[b, j+1], 2))
                probgridterm *= gauss_prob(obscolors[o, j], binmus[b, j+1], sig)
            sig = sqrt(pow(obsmags_err[o], 2) + pow(binsigs[b, 0], 2))
            for j in range(nbinsdist):
                probgrid[o, j] += probgridterm * gauss_prob(1/distance_grids[o, j], varpi[o], varpi_err[o]) * gauss_prob(5*log10(distance_grids[o, j]) + 10, obsmags[o] - binmus[b, 0], sig)


def prob_bingrid_distandbins_marg(
    double[:, :] probgrid,  # nobj, nbins
    long nobj, long nbins, long ncols,
    double[:] varpi,  # nobj
    double[:] varpi_err,  # nobj
    double[:] obsmags,  # nobj
    double[:, :] obscolors,  # nobj, ncols
    double[:] distances,  # nobj
    double[:] binamps,  # nbins
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :, :] sigmas  # nobj, nbins, ncols + 1
    ):
    cdef double valtot = 0, sig, val1, val2
    cdef long b, j, o
    for o in prange(nobj, nogil=True):
        val1 = gauss_prob(1/distances[o], varpi[o], varpi_err[o])
        val2 = 5*log10(distances[o]) + 10 - obsmags[o]
        for b in range(nbins):
            probgrid[o, b] *= binamps[b] * val1 * gauss_prob(val2,
                                                             -  binmus[b, 0],
                                                             sigmas[o, b, 0])


def prob_bingrid_magsonly_marg(
    double[:, :] probgrid,  # nobj, nbins
    long nobj, long nbins, long ncols,
    double[:] varpi,  # nobj
    double[:] varpi_err,  # nobj
    double[:] obsmags,  # nobj
    double[:, :] obscolors,  # nobj, ncols
    double[:] distances,  # nobj
    double[:] binamps,  # nbins
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :, :] sigmas  # nobj, nbins, ncols + 1
    ):
    cdef double valtot = 0, sig
    cdef long b, j, o
    for o in prange(nobj, nogil=True):
        for b in range(nbins):
            probgrid[o, b] = 1.0
            for j in range(ncols):
                probgrid[o, b] *= gauss_prob(obscolors[o, j],
                                             binmus[b, j+1],
                                             sigmas[o, b, j+1])


def sample_bins_marg(
    long[:] bins,  # nobj
    long[:] nearestbins,  # nobj
    long[:] counts,  # nobj
    long nobj, long nbins, long ncols,
    double[:] varpi,  # nobj
    double[:] varpi_err,  # nobj
    double[:] obsmags,  # nobj
    double[:, :] obscolors,  # nobj, ncols
    double[:] distances,  # nobj
    double[:] binamps,  # nbins
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :, :] sigmas  # nobj, nbins, ncols + 1
    ):
    cdef double sig, baseval, prob, x, disttobin, mindisttobin
    cdef long b, j, o, count, nearestbin
    cdef double *cumpdf
    cdef long *ibins
    cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

    for o in prange(nobj, nogil=True):
        cumpdf = <double *> malloc(sizeof(double) * (nbins+1))
        ibins = <long *> malloc(sizeof(long) * (nbins))
        if ibins == NULL or cumpdf == NULL:
            abort()
        baseval = gauss_prob(1/distances[o], varpi[o], varpi_err[o])
        count = 0
        mindisttobin = 1e160
        for b in range(nbins):
            disttobin = (obsmags[o] - 5*log10(distances[o]) - 10 - binmus[b, 0])**2
            prob = 1*baseval
            prob = prob * binamps[b] * gauss_prob(5*log10(distances[o]) + 10, obsmags[o] - binmus[b, 0], sigmas[o, b, 0])
            for j in range(ncols):
                prob = prob * gauss_prob(obscolors[o, j], binmus[b, j+1], sigmas[o, b, j+1])
                disttobin = disttobin + (obscolors[o, j] - binmus[b, j+1])**2
            if disttobin < mindisttobin:
                mindisttobin = disttobin
                nearestbin = b
            if prob > 1e-16:
                count = count + 1
                ibins[count-1] = b
                cumpdf[count] = cumpdf[count-1] + prob
        nearestbins[o] = nearestbin
        if count == 0:
            bins[o] = nearestbin
        if count == 1:
            bins[o] = ibins[count-1]
        if count > 1:
            x = uniform(r, 0, cumpdf[count])
            for b in range(1, count):
                if x < cumpdf[b] and x >= cumpdf[b-1]:
                    bins[o] = ibins[b-1]
        counts[o] = count
        free(cumpdf)
        free(ibins)


def prob_bingrid_marg(
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
    cdef double valtot = 0, sig, val
    cdef long b, j, o
    for o in prange(nobj, nogil=True):
        val = gauss_prob(1/distances[o], varpi[o], varpi_err[o])
        for b in range(nbins):
            probgrid[o, b] = val
            sig = sqrt(pow(obsmags_err[o], 2) + pow(binsigs[b, 0], 2))
            probgrid[o, b] *= binamps[b] * gauss_prob(5*log10(distances[o]) + 10, obsmags[o] - binmus[b, 0], sig)
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


cdef double snrcut_fac(double snr_lo, double snr_hi, double d, double sig) nogil:
    return 0.5 * (\
                    erf((1/d - sig*snr_lo) / sqrt(2) / sig) -\
                    erf((1/d - sig*snr_hi) / sqrt(2) / sig)
                    )


def lnprob_distgradient_marg_varpisnrcut(
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
    double[:, :] binsigs,  # nbins, ncols + 1
    double varpisnr_lo,
    double varpisnr_hi
    ):
    cdef double sig
    cdef long b, o
    for o in prange(nobj, nogil=True):
        distances_grad[o] = gaussdistvarpi_lnprob_grad_dist(distances[o], varpi[o], varpi_err[o])
        distances_grad[o] += \
            ( gauss_prob(1/distances[o], varpisnr_hi*varpi_err[o], varpi_err[o]) -\
              gauss_prob(1/distances[o], varpisnr_lo*varpi_err[o], varpi_err[o])
            ) / snrcut_fac(varpisnr_lo, varpisnr_hi, distances[o], varpi_err[o]) / pow(distances[o], 2)
        b = bins[o]
        sig = sqrt(pow(obsmags_err[o], 2) + pow(binsigs[b, 0], 2))
        distances_grad[o] +=\
            (5*log10(distances[o]) + 10 - obsmags[o] + binmus[b, 0])\
            * 5 / (sig*sig*distances[o]*log(10.))


def sample_bins_marg_varpisnrcut(
    long[:] bins,  # nobj
    long[:] nearestbins,  # nobj
    long[:] counts,  # nobj
    long nobj, long nbins, long ncols,
    double[:] varpi,  # nobj
    double[:] varpi_err,  # nobj
    double[:] obsmags,  # nobj
    double[:, :] obscolors,  # nobj, ncols
    double[:] distances,  # nobj
    double[:] binamps,  # nbins
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :, :] sigmas,  # nobj, nbins, ncols + 1
    double varpisnr_lo,
    double varpisnr_hi
    ):
    cdef double sig, baseval, prob, x, disttobin, mindisttobin
    cdef long b, j, o, count, nearestbin
    cdef double *cumpdf
    cdef long *ibins
    cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

    for o in prange(nobj, nogil=True):
        cumpdf = <double *> malloc(sizeof(double) * (nbins+1))
        ibins = <long *> malloc(sizeof(long) * (nbins))
        if ibins == NULL or cumpdf == NULL:
            abort()
        baseval = gauss_prob(1/distances[o], varpi[o], varpi_err[o]) /\
            snrcut_fac(varpisnr_lo, varpisnr_hi, distances[o], varpi_err[o])
        count = 0
        mindisttobin = 1e160
        for b in range(nbins):
            disttobin = (obsmags[o] - 5*log10(distances[o]) - 10 - binmus[b, 0])**2
            prob = 1*baseval
            prob = prob * binamps[b] * gauss_prob(5*log10(distances[o]) + 10, obsmags[o] - binmus[b, 0], sigmas[o, b, 0])
            for j in range(ncols):
                prob = prob * gauss_prob(obscolors[o, j], binmus[b, j+1], sigmas[o, b, j+1])
                disttobin = disttobin + (obscolors[o, j] - binmus[b, j+1])**2
            if disttobin < mindisttobin:
                mindisttobin = disttobin
                nearestbin = b
            if prob > 1e-12:
                count = count + 1
                ibins[count-1] = b
                cumpdf[count] = cumpdf[count-1] + prob
        nearestbins[o] = nearestbin
        if count == 0:
            bins[o] = nearestbin
        if count == 1:
            bins[o] = ibins[count-1]
        if count > 1:
            x = uniform(r, 0, cumpdf[count])
            for b in range(1, count):
                if x < cumpdf[b] and x >= cumpdf[b-1]:
                    bins[o] = ibins[b-1]
        counts[o] = count
        free(cumpdf)
        free(ibins)


def lnprob_marg_varpisnrcut(
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
    double[:, :] binsigs,  # nbins, ncols + 1
    double varpisnr_lo,
    double varpisnr_hi
    ):
    cdef double valtot = 0, sig
    cdef long b, j, o
    for o in prange(nobj, nogil=True):
        b = bins[o]
        valtot += - log(binamps[b])
        valtot += gaussdistvarpi_lnprob(distances[o], varpi[o], varpi_err[o])
        valtot += log(snrcut_fac(varpisnr_lo, varpisnr_hi, distances[o], varpi_err[o]))
        sig = sqrt(pow(obsmags_err[o], 2) + pow(binsigs[b, 0], 2))
        valtot += gauss_lnprob(5.*log10(distances[o]) + 10.,
                               obsmags[o] - binmus[b, 0], sig)
        for j in range(ncols):
            sig = sqrt(pow(obscolors_err[o, j], 2) + pow(binsigs[b, j+1], 2))
            valtot += gauss_lnprob(obscolors[o, j], binmus[b, j+1], sig)
    return valtot


def prob_distgrid_marg_varpisnrcut(
    double[:, :] probgrid,  # nobj, nbins
    long nbinsdist, double[:] distances_grid,  # nobj
    long nobj, long nbins, long ncols,
    double[:] varpi,  # nobj
    double[:] varpi_err,  # nobj
    double[:] obsmags,  # nobj
    double[:] obsmags_err,  # nobj
    double[:, :] obscolors,  # nobj, ncols
    double[:, :] obscolors_err,  # nobj, ncols
    double[:] binamps,  # nbins
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :] binsigs,  # nbins, ncols + 1
    double varpisnr_lo,
    double varpisnr_hi
    ):
    cdef double valtot = 0, sig, probgridterm
    cdef long b, j, o
    for o in prange(nobj, nogil=True):
        for b in range(nbins):
            probgridterm = binamps[b]
            for j in range(ncols):
                sig = sqrt(pow(obscolors_err[o, j], 2) + pow(binsigs[b, j+1], 2))
                probgridterm *= gauss_prob(obscolors[o, j], binmus[b, j+1], sig)
            sig = sqrt(pow(obsmags_err[o], 2) + pow(binsigs[b, 0], 2))
            for j in range(nbinsdist):
                probgrid[o, j] += probgridterm * gauss_prob(1/distances_grid[j], varpi[o], varpi_err[o]) * gauss_prob(5*log10(distances_grid[j]) + 10, obsmags[o] - binmus[b, 0], sig) / snrcut_fac(varpisnr_lo, varpisnr_hi, distances_grid[j], varpi_err[o])


def prob_bingrid_fullmarg(
    double[:, :] probgrid,  # nobj, nbins
    double dist_min, double dist_max,
    long nobj, long nbins, long ncols,
    double[:] varpi,  # nobj
    double[:] varpi_err,  # nobj
    double[:] obsmags,  # nobj
    double[:] obsmags_err,  # nobj
    double[:, :] obscolors,  # nobj, ncols
    double[:, :] obscolors_err,  # nobj, ncols
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :] binsigs
    ):
    cdef long numpts = 400
    cdef long fac = 4
    cdef double valtot = 0, sig, probgridterm
    cdef double delta_d, mud, sigd, hes, d_min, d_max, d_val, dist_err
    cdef double M1, M2, S1, S2, mutot
    cdef long b, j, o
    for o in prange(nobj, nogil=True):
        dist_err = varpi_err[o]/pow(varpi[o], 2)
        for b in range(nbins):
            probgridterm = 1.0
            for j in range(ncols):
                sig = sqrt(pow(obscolors_err[o, j], 2) + pow(binsigs[b, j+1], 2))
                probgridterm *= gauss_prob(obscolors[o, j], binmus[b, j+1], sig)
            sig = sqrt(pow(obsmags_err[o], 2) + pow(binsigs[b, 0], 2))
            mud = pow(10, -0.2*(binmus[b, 0] - obsmags[o] + 10)) - 1/varpi[o]
            hes = pow(5/log(10)/mud/sig, 2) - (5*log10(mud) - obsmags[o] + binmus[b, 0] + 10) / pow(sig*mud, 2) * (5/log(10))
            M1 = mud
            S1 = pow(hes, -0.5)
            M2 = 1./varpi[o]
            S2 = pow(dist_err, 2.)
            mutot = (M1*S2 + M2*S1) / (S1 + S2)
            sigd = sqrt(S1*S2 / (S1 + S2))
            d_min = max(dist_min, mutot - fac*sigd)
            d_max = min(dist_max, mutot + fac*sigd)
            delta_d = (d_max - d_min) / float(numpts-1)
            for j in range(numpts):
                d_val = d_min + j * delta_d
                probgrid[o, b] += probgridterm * delta_d * gauss_prob(1/d_val, varpi[o], varpi_err[o]) * gauss_prob(5*log10(d_val) + 10, obsmags[o] - binmus[b, 0], sig)


def sample_bins_from_grid(
    long[:] bins,  # nobj
    double[:, :] probgrid,  # nobj, nbins
    double[:] binamps, # nbins
    long nobj, long nbins
    ):
    cdef double sig, x
    cdef long b, j, o
    cdef double *cumpdf
    cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

    for o in prange(nobj, nogil=True):
        cumpdf = <double *> malloc(sizeof(double) * (nbins+1))
        if cumpdf == NULL:
            abort()
        for b in range(nbins):
            cumpdf[b+1] = cumpdf[b] + probgrid[o, b] * binamps[b]
        x = uniform(r, 0, cumpdf[nbins])
        for b in range(1, nbins + 1):
            if x < cumpdf[b] and x >= cumpdf[b-1]:
                bins[o] = b - 1
        free(cumpdf)


cdef double logsumexp(double* arr, long dim) nogil:
    cdef int i
    cdef double result = 0.0
    cdef double largest_in_a = arr[0]
    for i in range(1, dim):
        if (arr[i] > largest_in_a):
            largest_in_a = arr[i]
    for i in range(dim):
        result += exp(arr[i] - largest_in_a)
    return largest_in_a + log(result)


cdef double univariate_normal_lnprob(double x, double mu, double var) nogil:
    return - 0.5 * pow(x - mu, 2)/ var - 0.5 * log(2*M_PI) - 0.5 * log(var)

cdef double bivariate_normal_lnprob(double x1, double x2, 
                                    double mu1, double mu2, 
                                    double var1, double var2, double rho) nogil:
    cdef double z = pow(x1 - mu1, 2.) / var1 + pow(x2 - mu2, 2.) / var2\
        - 2 * rho * (x1 - mu1) * (x2 - mu2) / pow(var1 * var2, 0.5)
    return - 0.5 * z / (1. - rho*rho) - log(2*M_PI)\
         - 0.5 * log(var1 * var2 * (1. - rho*rho)) 

    
cdef create_covariances(
    double[:, :, :] bincovars, # nbins, (ncols + 1), (ncols + 1)
    double[:, :] binrhos, # nbins, (ncols - 1)*ncols//2
    double[:, :] binsigs, # nbins, (ncols + 1)
    long nbins, long ncols
                       ):
    cdef long b, i, j
    for b in range(nbins):
        for i in range(ncols+1):
            for j in range(i):
                bincovars[b, i*(i+1) / 2 + j] = binrhos[b, i*(i-1)//2 + j] * binsigs[b, i] * binsigs[b, j]
            bincovars[b, i*(i+1) / 2 + i] = binsigs[b, i] * binsigs[b, i]


def lnprob_dustdistbinmarg(
    long nobj, long nbins, long nmarggrid, double sigma,
    double[:] obsvarpis,  # nobj
    double[:] obsvarpis_var,  # nobj
    double[:] obsmags,  # nobj
    double[:] obsmags_var,  # nobj
    double[:] obscolors,  # nobj, ncols
    double[:] obscolors_var,  # nobj, ncols
    double[:] dustpriormeans,  # nobj
    double[:] dustpriorvars,  # nobj
    double[:] dustcoefs,  # ncols + 1
    double[:] binamps,  # nbins
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :] binvars,  # nbins, ncols
    double[:, :] binrhos  # nbins, (ncols - 1) * (ncols) // 2
     ):

    cdef double lnprobtot = 0
    cdef double* probgrid
    cdef long b, i, j, o, kv, kd
    cdef double deltavarpi, deltadust, varpi, dustamp, varpi_min, varpi_max, binrho
    
    for o in prange(nobj, nogil=True):
        
        probgrid = <double * > malloc(sizeof(double) * (nbins * nmarggrid * nmarggrid))
        
        for b in range(nbins):
            varpi_min = max(1e-4, obsvarpis[o] - sigma * pow(obsvarpis_var[o], 0.5))
            varpi_max = obsvarpis[o] + sigma * pow(obsvarpis_var[o], 0.5)
            deltavarpi =  (varpi_max - varpi_min) / nmarggrid
            deltadust = 2 * sigma * pow(dustpriorvars[o], 0.5) / nmarggrid
            binrho = binrhos[b, 0] * sqrt(binvars[b, 0] * binvars[b, 1]) / sqrt((obsmags_var[o] + binvars[b, 0])*(obscolors_var[o] + binvars[b, 1]))
            for kv in range(nmarggrid):
                varpi = varpi_min + kv * deltavarpi
                for kd in range(nmarggrid):
                    dustamp = dustpriormeans[o] - sigma * pow(dustpriorvars[o], 0.5) + kd * deltadust
                    probgrid[b*nmarggrid*nmarggrid + kv*nmarggrid + kd] = \
                        log(deltadust * deltavarpi * binamps[b])\
                        + univariate_normal_lnprob(obsvarpis[o], varpi, obsvarpis_var[o])\
                        + univariate_normal_lnprob(dustamp, dustpriormeans[o], dustpriorvars[o])\
                        + bivariate_normal_lnprob(
                              binmus[b, 0], 
                              binmus[b, 1],
                              obsmags[o] - dustamp * dustcoefs[0] + 5*log10(varpi) - 10,
                              obscolors[o] - dustamp * dustcoefs[1], 
                              obsmags_var[o] + binvars[b, 0],
                              obscolors_var[o] + binvars[b, 1],
                              binrho
                        )

        lnprobtot += logsumexp(probgrid, nbins*nmarggrid*nmarggrid)

        free(probgrid)

    return lnprobtot


cdef double matrix3x3_determinant(double* a) nogil:
    return (a[0*3+0] * (a[1*3+1] * a[2*3+2] - a[2*3+1] * a[1*3+2])\
           -a[1*3+0] * (a[0*3+1] * a[2*3+2] - a[2*3+1] * a[0*3+2])\
           +a[2*3+0] * (a[0*3+1] * a[1*3+2] - a[1*3+1] * a[0*3+2]))


cdef double matrix3x3_quadform(
        double x1, double x2, double x3, 
        double* a, double det) nogil:
    cdef double b11, b12, b13, b22, b23, b33
    b11 = a[1*3+1] * a[2*3+2] - a[1*3+2] * a[2*3+1]
    b12 = a[0*3+2] * a[2*3+1] - a[0*3+1] * a[2*3+2]
    b13 = a[0*3+1] * a[1*3+2] - a[0*3+2] * a[1*3+1]
    b22 = a[0*3+0] * a[2*3+2] - a[0*3+2] * a[2*3+0]
    b23 = a[0*3+2] * a[1*3+0] - a[0*3+0] * a[1*3+2]
    b33 = a[0*3+0] * a[1*3+1] - a[0*3+1] * a[1*3+0]
    return (b11*x1*x1 + 2*b12*x1*x2 + 2*b13*x1*x3 +\
             b22*x2*x2 + 2*b23*x2*x3 + b33*x3*x3) / det


def varpipm_likelihood_velocitybinmarg(
        double[:, :] probgrid,
        long nbins, 
        long nobj,
        double varpi_min, 
        double varpi_max,
        long nmarggrid,
        double[:] obsvarpis,  # data (point estimates)
        double[:, :, :] xyz2radec, 
        double[:, :, :] mu_varpi_covars,  # data (covariance of the estimates)
        double[:] pm_ras, 
        double[:] pm_decs,
        double[:] vxyz_amps, 
        double[:, :] vxyz_mus, 
        double[:, :, :] vxyz_covars
        ):

    cdef double lnprobtot = 0
    #cdef double* probgrid
    cdef long b, i, j, o, kv, k1, k2
    cdef double deltavarpi, varpi, delta1, delta2, delta3, det #varpi_min, varpi_max, 
    
    for o in prange(nobj, nogil=True):#range(nobj):#

        comp_covar = <double * > malloc(sizeof(double) * (3 * 3))
        
        for b in range(nbins):
            deltavarpi =  (varpi_max - varpi_min) / (nmarggrid - 1)

            for kv in range(nmarggrid):
                varpi = varpi_min + kv * deltavarpi

                delta1 = - pm_ras[o]
                delta2 = - pm_decs[o]
                delta3 = varpi - obsvarpis[o]
                for k1 in range(3):
                    delta1 = delta1 + varpi * xyz2radec[o, 0, k1] * vxyz_mus[b, k1] 
                    delta2 = delta2 + varpi * xyz2radec[o, 1, k1] * vxyz_mus[b, k1]
                
                for i in range(3):
                    for j in range(3):
                            comp_covar[i*3 + j] = mu_varpi_covars[o, i, j]

                for i in range(2):
                    for j in range(2):
                        for k1 in range(3):
                            for k2 in range(3):
                                 comp_covar[i*3 + j] += varpi**2. * \
                                    xyz2radec[o, i, k1] * vxyz_covars[b, k1, k2] *\
                                         xyz2radec[o, j, k2]

                det = matrix3x3_determinant(comp_covar)
               
                probgrid[o, kv] += vxyz_amps[b] * exp(\
                    - 0.5 * log(det) - 0.5 * matrix3x3_quadform(delta1, delta2, delta3, comp_covar, det))

        free(comp_covar)

    return lnprobtot





def varpipm_likelihood_velocityvarpibinmarg(
        long nbins, 
        long nobj,
        long nmarggrid,
        double sigma,
        double[:] obsvarpis, 
        double[:] obsvarpis_err, 
        double[:, :, :] xyz2radec, 
        double[:, :, :] mu_varpi_covars, 
        double[:] pm_ras, 
        double[:] pm_decs,
        double[:] vxyz_amps, 
        double[:, :] vxyz_mus, 
        double[:, :, :] vxyz_covars
        ):

    cdef double lnprobtot = 0
    #cdef double* probgrid
    cdef long b, i, j, o, kv, k1, k2
    cdef double deltavarpi, varpi, delta1, delta2, delta3, det #varpi_min, varpi_max, 
    
    for o in range(nobj):#prange(nobj, nogil=True):#range(nobj):#

        probgrid = <double * > malloc(sizeof(double) * (nbins * nmarggrid))
        comp_covar = <double * > malloc(sizeof(double) * (3 * 3))
        
        for b in range(nbins):
            varpi_min = max(1e-4, obsvarpis[o] - sigma * obsvarpis_err[o])
            varpi_max = obsvarpis[o] + sigma * obsvarpis_err[o]
            deltavarpi =  (varpi_max - varpi_min) / (nmarggrid - 1)

            for kv in range(nmarggrid):
                varpi = varpi_min + kv * deltavarpi

                delta1 = - pm_ras[o]
                delta2 = - pm_decs[o]
                delta3 = varpi - obsvarpis[o]
                for k1 in range(3):
                    delta1 = delta1 + varpi * xyz2radec[o, 0, k1] * vxyz_mus[b, k1] 
                    delta2 = delta2 + varpi * xyz2radec[o, 1, k1] * vxyz_mus[b, k1]
                
                for i in range(3):
                    for j in range(3):
                            comp_covar[i*3 + j] = mu_varpi_covars[o, i, j]

                for i in range(2):
                    for j in range(2):
                        for k1 in range(3):
                            for k2 in range(3):
                                 comp_covar[i*3 + j] += varpi**2. * \
                                    xyz2radec[o, i, k1] * vxyz_covars[b, k1, k2] *\
                                         xyz2radec[o, j, k2]

                det = matrix3x3_determinant(comp_covar)

                probgrid[b*nmarggrid + kv] = \
                    log(deltavarpi * vxyz_amps[b])\
                    - 0.5 * log(det) - 0.5 * matrix3x3_quadform(delta1, delta2, delta3, comp_covar, det)

        lnprobtot += logsumexp(probgrid, nbins*nmarggrid)

        free(probgrid)
        free(comp_covar)


    return lnprobtot


cdef double multivariate_normal_chi2(int ndim, double * mean, double * covar_chol) nogil:
    cdef int i, rhs = 1, info
    temp = <double * > malloc(sizeof(double) * (ndim))
    for i in range(ndim):
        temp[i] = mean[i]
    cython_lapack.dpotrs( 'U', &ndim, &rhs, covar_chol, &ndim, temp, &ndim, &info)
    chi2 = 0.0
    for i in range(ndim):
        chi2 += temp[i] * mean[i]
    free(temp)
    return chi2


def lnprob_multicolor_distbinmarg(
    long nobj, long ncols, long nbins, long nmarggrid, double sigma,
    double[:] obsvarpis,  # nobj
    double[:] obsvarpis_var,  # nobj
    double[:] obsmags,  # nobj
    double[:] obsmags_var,  # nobj
    double[:, :] obscolors,  # nobj, ncols
    double[:, :] obscolors_var,  # nobj, ncols
    double[:] dustamps,  # nobj
    double[:] dustcoefs,  # ncols + 1
    double[:] binamps,  # nbins
    double[:, :] binmus,  # nbins, ncols + 1
    double[:, :, :] bincovars  # nbins, (ncols + 1)*(ncols + 1)
     ):

    cdef double lnprobtot = 0
    cdef double* probgrid
    cdef long b, i, j, o, kv
    cdef double deltavarpi, varpi, varpi_min, varpi_max, det
    cdef int ndim = ncols + 1, info
    
    for o in prange(nobj, nogil=True):#range(nobj):#
        
        probgrid = <double * > malloc(sizeof(double) * (nbins * nmarggrid))
        meanvec = <double * > malloc(sizeof(double) * (ncols+1))
        covar = <double * > malloc(sizeof(double) * ((ncols+1) * (ncols+1)))

        varpi_min = max(1e-4, obsvarpis[o] - sigma * pow(obsvarpis_var[o], 0.5))
        varpi_max = obsvarpis[o] + sigma * pow(obsvarpis_var[o], 0.5)
        deltavarpi =  (varpi_max - varpi_min) / nmarggrid
        
        for b in range(nbins):
            
            for i in range(ncols+1):
                for j in range(ncols+1):
                    covar[i*(ncols+1) + j] = bincovars[b, i, j]

            covar[0] = covar[0] + obsmags_var[o]
            for i in range(ncols):
                meanvec[1+i] = obscolors[o, i] - dustamps[o] * dustcoefs[1+i] - binmus[b, i+1]
                covar[(1+i)*(ncols+1) + (1+i)] = covar[(1+i)*(ncols+1) + (1+i)] + obscolors_var[o, i]

            cython_lapack.dpotrf('U', &ndim, covar, &ndim, &info)

            det = 1.0
            for i in range(ncols+1):
                det = det * pow(covar[i*(ncols+1)+i], 2.)

            for kv in range(nmarggrid):
                varpi = varpi_min + kv * deltavarpi
                meanvec[0] = obsmags[o] - dustamps[o] * dustcoefs[0] + 5*log10(varpi) - 10 - binmus[b, 0]
                probgrid[b*nmarggrid + kv] = \
                    log(deltavarpi * binamps[b])\
                    + univariate_normal_lnprob(obsvarpis[o], varpi, obsvarpis_var[o])\
                    - 0.5 * multivariate_normal_chi2(ndim, meanvec, covar) - 0.5 * log(det) - 0.5 * ndim * log(2*M_PI) 

        lnprobtot += logsumexp(probgrid, nbins*nmarggrid)

        free(probgrid)
        free(meanvec)
        free(covar)

    return lnprobtot

         
def lnprob_distbinmarg(
    long nobj, # number of objects 
    long nbins, # number of bins of the Gaussian mixture 
    long nmarggrid, # number of points for the numerical marginalization of the parallax.
    double sigma, # for each object, the marginalization grid will be nmarggrid points between [obsvarpis - sigma*obsvarpis_var**0.5, obsvarpis + sigma*obsvarpis_var**0.5]
    double[:] obsvarpis,  # Observed parallaxes. Array of size nobj 
    double[:] obsvarpis_var,  # Parallax Variances. Array of size nobj
    double[:] obsmags,  # Observed magnitudes. Array of size nobj
    double[:] obsmags_var,  # Variance of obs magnitudes. Array of size nobj
    double[:] obscolors,  # Observed color. Array of size nobj
    double[:] obscolors_var,  # Variance of colors. Array of size  nobj
    double[:] dustamps,  # Amplitudes of the dust for each object. Array of size nobj 
    double[:] dustcoefs,  # Dust coefficients for the magnitude and the color. Array of size 2
    double[:] binamps,  # Normalized amplitudes of the Gaussian mixture components. Array of size nbins
    double[:, :] binmus,  # Positions of the Gaussian mixture components. Array of size nbins x 2
    double[:, :] binvars,  # Variances of the Gaussian mixture components. Array of size nbins x 2
    double[:, :] binrhos  # Correlation coefficients of the Gaussian mixture components. Array of size nbins x 1
     ):

    cdef double lnprobtot = 0 # total log posterior probability
    cdef double* probgrid # nbins * nmarggrid grid instanciated for each object, to compute the posterior probability and marginalize over the bins and parallaxes.
    cdef long b, i, j, o, kv
    cdef double deltavarpi, varpi, varpi_min, varpi_max, binrho
    # deltavarpi is the separation of varpis on the grid of each object.
    # varpi is a temp variable for the running parallax
    # varpi_min, varpi_max will be the bounds of the grid for each object
    # binrho is the correlation coefficient for the bth bin and the oth object, once true color and mag are marginalized over.
    
    # parallel loop over o
    for o in prange(nobj, nogil=True):
        
        # create the grid
        probgrid = <double * > malloc(sizeof(double) * (nbins * nmarggrid))

        # compute the bounds and resolution of the grid for the object.
        varpi_min = max(1e-4, obsvarpis[o] - sigma * pow(obsvarpis_var[o], 0.5))
        varpi_max = obsvarpis[o] + sigma * pow(obsvarpis_var[o], 0.5)
        deltavarpi =  (varpi_max - varpi_min) / (nmarggrid - 1)
        
        # loop over bins
        for b in range(nbins):
        	# compute the corr coefficient once the mixture and observed errors are added to marginalize over mag and color.
            binrho = binrhos[b, 0] * sqrt(binvars[b, 0] * binvars[b, 1]) / sqrt((obsmags_var[o] + binvars[b, 0])*(obscolors_var[o] + binvars[b, 1]))
            # loop over parallax grid.
            for kv in range(nmarggrid):
                varpi = varpi_min + kv * deltavarpi # the actual parallax we are considering
                # compute and store the log posterior distribution for that object, bin, and parallax value.
                probgrid[b*nmarggrid + kv] = \
                    log(deltavarpi * binamps[b])\
                    + univariate_normal_lnprob(obsvarpis[o], varpi, obsvarpis_var[o])\
                    + bivariate_normal_lnprob(
                          binmus[b, 0], 
                          binmus[b, 1],
                          obsmags[o] - dustamps[o] * dustcoefs[0] + 5*log10(varpi) - 10,
                          obscolors[o] - dustamps[o] * dustcoefs[1], 
                          obsmags_var[o] + binvars[b, 0],
                          obscolors_var[o] + binvars[b, 1],
                          binrho
                    )

        # perform the log sum exp of the grid, to get the log posterior value for this object with bin and parallax marginalized over. Then sum them all.
        lnprobtot += logsumexp(probgrid, nbins*nmarggrid)

        # free the grid
        free(probgrid)

    return lnprobtot

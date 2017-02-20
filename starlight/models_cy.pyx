#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
cimport numpy as np
import numpy as np
from cython.parallel import prange
from cpython cimport bool
cimport cython
from libc.math cimport sqrt, M_PI, exp, pow, erf, log, log10
from libc.stdlib cimport abort, malloc, free

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
    cdef long numpts = 100
    cdef long fac = 4
    cdef double valtot = 0, sig, probgridterm
    cdef double delta_d, mud, sigd, hes, d_min, d_max, d_val
    cdef long b, j, o
    for o in prange(nobj, nogil=True):
        for b in range(nbins):
            probgridterm = 1.0
            for j in range(ncols):
                sig = sqrt(pow(obscolors_err[o, j], 2) + pow(binsigs[b, j+1], 2))
                probgridterm *= gauss_prob(obscolors[o, j], binmus[b, j+1], sig)
            sig = sqrt(pow(obsmags_err[o], 2) + pow(binsigs[b, 0], 2))
            mud = pow(10, -0.2*(binmus[b, 0] - obsmags[o] + 10))
            hes = pow(5/log(10)/mud/sig, 2) - (5*log10(mud) - obsmags[o] + binmus[b, 0] + 10) / pow(sig*mud, 2) * (5/log(10))
            sigd = pow(hes, -0.5)
            if (mud - 1/varpi[o]) < varpi_err[o]/pow(varpi[o], 2) + fac*sigd:
                d_min = max(dist_min, mud - fac*sigd)
                d_max = min(dist_max, mud + fac*sigd)
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

#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
cimport numpy as np
import numpy as np
from cython.parallel import prange
from cpython cimport bool
cimport cython
from libc.math cimport sqrt, M_PI, exp, pow, erf, log10
from libc.stdlib cimport abort, malloc, free


cdef extern from "gsl/gsl_rng.h":
	ctypedef struct gsl_rng_type:
		pass
	ctypedef struct gsl_rng:
		pass
	gsl_rng_type *gsl_rng_mt19937
	gsl_rng *gsl_rng_alloc(gsl_rng_type * T)

cdef extern from "gsl/gsl_randist.h":
	double gamma "gsl_ran_gamma"(gsl_rng * r,double,double) nogil
	double gaussian "gsl_ran_gaussian"(gsl_rng * r,double) nogil
	double uniform "gsl_ran_flat"(const gsl_rng * r, double, double) nogil


def gibbs_sampler(long nsamples, 
	double[:, :] mags,
	double[:, :] mags_err, 
	double[:] varpi,
	double[:] varpi_err, 
	double[:, :] grids_min, 
	double[:, :] grids_max, 
	double[:] nbs_ini):

	cdef double sqrt2pi = sqrt(2. / M_PI)
	cdef double sqrt2 = sqrt(2.)

	cdef long b, i, o, kk
	cdef long nobj = mags.shape[0]
	cdef long ndim = mags.shape[1]
	cdef long nbins = nbs_ini.size
	cdef np.ndarray[double, ndim=1] hbs = np.zeros(nbins)
	cdef np.ndarray[double, ndim=2] fbs = np.zeros((nsamples, nbins))
	cdef np.ndarray[double, ndim=2] dist = np.zeros((nsamples, nobj))
	cdef np.ndarray[double, ndim=2] sigmas = np.zeros((nobj, nbins))
	cdef np.ndarray[double, ndim=1] nbs = np.zeros(nbins)
	cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

	cdef double pos, Mi, cumsum
	cdef double *cumsumweights, *pdfint

	cumsum = 0
	for b in range(nbins):
		nbs[b] = nbs_ini[b]
		hbs[b] = gamma(r, nbs[b] + 1, 1)
		cumsum += hbs[b]
	for b in range(nbins):
		hbs[b] = hbs[b] / cumsum

	for o in prange(nobj, nogil=True):
		for i in range(ndim):
			if i == 0:
				sigmas[o, i] = mags_err[o, 0]
			else:
				sigmas[o, i] = sqrt(pow(mags_err[o, 0], 2) + pow(mags_err[o, i], 2))

	for kk in range(1, nsamples):
		for b in range(nbins):
			nbs[b] = 0

		for o in prange(nobj, nogil=True):

			pdfint = <double *> malloc(sizeof(double) * nbins)
			cumsumweights = <double *> malloc(sizeof(double) * nbins)
			if pdfint == NULL or cumsumweights == NULL:
				abort()
			for b in range(nbins):
				pdfint[b] = 1

			dist[kk, o] = 1. / (varpi[o] + gaussian(r, varpi_err[o]))
			for i in range(ndim):
				if i == 0:
					Mi = (mags[o, 0] - 5*log10(dist[kk, o]) - 10)
				else:
					Mi = (mags[o, i] - mags[o, 0])
				for b in range(nbins):
					pdfint[b] *= sqrt2pi * sigmas[o, i] * ( erf((Mi - grids_min[i, b])/sigmas[o, i]/sqrt2) - erf((Mi - grids_max[i, b])/sigmas[o, i]/sqrt2) )

			for b in range(nbins):
				pdfint[b] *= hbs[b]

			cumsumweights[0] = 0
			for b in range(1, nbins):
				cumsumweights[b] = cumsumweights[b-1] + pdfint[b]

			pos = uniform(r, 0, cumsumweights[nbins-1])
			for b in range(nbins):
				if(pos > cumsumweights[b]) and (pos <= cumsumweights[b+1]):
					nbs[b] += 1
					break

			free(cumsumweights)
			free(pdfint)

		cumsum = 0
		for b in range(nbins):
			hbs[b] = gamma(r, nbs[b] + 1, 1)
			cumsum += hbs[b]
		for b in range(nbins):
			hbs[b] = hbs[b] / cumsum
		for b in range(nbins):
			fbs[kk, b] = hbs[b]

	return fbs, dist

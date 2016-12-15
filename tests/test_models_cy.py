
import numpy as np
from scipy.misc import derivative
import pytest

from starlight.models_cy import *

relative_accuracy = 0.0001
NREPEAT = 10


def gaussian(x, mu, sig):
    return np.exp(-0.5*((x - mu)/sig)**2) / np.sqrt(2*np.pi) / sig


def gaussian_grad(x, mu, sig):
    return - gaussian(x, mu, sig) * (x - mu) / sig**2


def lngaussian(x, mu, sig):
    return 0.5*((x-mu)/sig)**2 + 0.5*np.log(2*np.pi) + np.log(sig)


def lngaussian_grad(x, mu, sig):
    return (x-mu)/sig**2


def mylnprob_and_grads_nomarg(
    nobj, nbins, ncols,
    varpi, varpi_err,  # nobj
    obsmags, obsmags_err,  # nobj
    obscolors, obscolors_err,  # nobj, ncols
    absmags, distances,  # nobj
    colors,  # nobj, ncols
    binamps,  # nbins
    binmus,  # nbins, ncols + 1
    binsigs  # nbins, ncols + 1
        ):

    lnprobval = np.sum(
        lngaussian(1/distances, varpi, varpi_err) +
        lngaussian(absmags + 5*np.log10(distances) + 10, obsmags, obsmags_err)
        )
    binprobs = binamps[None, :] * gaussian(
            absmags[:, None], binmus[None, :, 0], binsigs[None, :, 0])
    for i in range(ncols):
        lnprobval += np.sum(
            lngaussian(colors[:, i], obscolors[:, i], obscolors_err[:, i])
            )
        binprobs *= gaussian(
                colors[:, i, None],
                binmus[None, :, i + 1],
                binsigs[None, :, i + 1])
    binlnprobtot = - np.log(binprobs.sum(axis=1)).sum()
    lnprobval += binlnprobtot

    oldterm = binamps[None, :] * gaussian(
            absmags[:, None], binmus[None, :, 0], binsigs[None, :, 0])
    newterm = binamps[None, :] * gaussian_grad(
            absmags[:, None], binmus[None, :, 0], binsigs[None, :, 0])
    absmags_grad =\
        - (binprobs * newterm / oldterm).sum(axis=1) / binprobs.sum(axis=1)
    colors_grad = np.zeros((nobj, ncols))
    for i in range(ncols):
        colors_grad[:, i] = lngaussian_grad(colors[:, i],
                                            obscolors[:, i],
                                            obscolors_err[:, i])
        oldterm = binamps[None, :] * gaussian(
                colors[:, i, None],
                binmus[None, :, i + 1],
                binsigs[None, :, i + 1])
        newterm = binamps[None, :] * gaussian_grad(
                colors[:, i, None],
                binmus[None, :, i + 1],
                binsigs[None, :, i + 1])
        colors_grad[:, i] +=\
            - (binprobs * newterm / oldterm).sum(axis=1) / binprobs.sum(axis=1)

    binamps_grad = - np.sum(
        binprobs[:, :] / binamps[None, :] / binprobs.sum(axis=1)[:, None],
        axis=0)

    absmags_grad += (absmags+5*np.log10(distances)+10-obsmags) / obsmags_err**2
    distances_grad = 5 * (absmags+5*np.log10(distances)+10 - obsmags) /\
        (obsmags_err**2 * distances * np.log(10))
    distances_grad += - (1/distances - varpi) / (varpi_err * distances)**2

    return lnprobval, absmags_grad, distances_grad, colors_grad, binamps_grad


@pytest.mark.skip(reason="Annoying tests")
def test_SimpleHDRModel_nomarg_gradients():

    for k in range(NREPEAT):

        nbins = np.random.randint(4, 100)
        nobj = np.random.randint(10, 100)
        ncols = np.random.randint(1, 3)

        absmags = np.random.uniform(1, 2, nobj)
        distances = np.random.uniform(0.1, 0.3, nobj)
        varpi = 1/distances
        varpi_err = varpi*0.01
        varpi += varpi_err*np.random.randn(*varpi.shape)
        colors = np.random.uniform(1, 2, nobj*ncols).reshape((nobj, ncols))
        binamps = np.random.uniform(0, 1, nbins)
        binmus = np.random.uniform(1, 2, nbins*(ncols+1))\
            .reshape((nbins, ncols+1))
        binsigs = np.repeat(0.5, nbins*(ncols+1)).reshape((nbins, ncols+1))
        obsmags = absmags + 5*np.log10(distances) + 10
        obsmags_err = obsmags*0.01
        obsmags += obsmags_err * np.random.randn(*obsmags.shape)
        obscolors = 1*colors
        obscolors_err = obscolors*0.01
        obscolors += obscolors_err*np.random.randn(*colors.shape)

        lnprobval2, absmags_grad2, distances_grad2,\
            colors_grad2, binamps_grad2 =\
            mylnprob_and_grads_nomarg(
                nobj, nbins, ncols, varpi, varpi_err,
                obsmags, obsmags_err, obscolors, obscolors_err,
                absmags, distances, colors, binamps, binmus, binsigs)

        lnprobval1 = lnprob_nomarg(
            nobj, nbins, ncols, varpi, varpi_err,
            obsmags, obsmags_err, obscolors, obscolors_err,
            absmags, distances, colors, binamps, binmus, binsigs)

        assert np.abs(lnprobval2/lnprobval1 - 1 < relative_accuracy)

        absmags_grad1, distances_grad1, colors_grad1, binamps_grad1 =\
            0*absmags_grad2, 0*distances_grad2, 0*colors_grad2, 0*binamps_grad2

        lnprob_gradients_nomarg(
            absmags_grad1, distances_grad1, colors_grad1, binamps_grad1,
            nobj, nbins, ncols, varpi, varpi_err,
            obsmags, obsmags_err, obscolors, obscolors_err,
            absmags, distances, colors, binamps, binmus, binsigs)

        np.testing.assert_allclose(distances_grad1, distances_grad2,
                                   rtol=relative_accuracy)
        for i in range(nobj):
            def f(d):
                distances2 = 1*distances
                distances2[i] = d
                return lnprob_nomarg(
                    nobj, nbins, ncols, varpi, varpi_err,
                    obsmags, obsmags_err, obscolors, obscolors_err,
                    absmags, distances2, colors, binamps, binmus, binsigs)

            distances_grad3 = derivative(f, 1*distances[i],
                                         dx=0.001*distances[i], order=5)
            assert abs(distances_grad3/distances_grad2[i] - 1)\
                < relative_accuracy

        np.testing.assert_allclose(absmags_grad1, absmags_grad2,
                                   rtol=relative_accuracy)
        for i in range(nobj):
            def f(d):
                absmags2 = 1*absmags
                absmags2[i] = d
                return lnprob_nomarg(
                    nobj, nbins, ncols, varpi, varpi_err,
                    obsmags, obsmags_err, obscolors, obscolors_err,
                    absmags2, distances, colors, binamps, binmus, binsigs)
            absmags_grad3 = derivative(f, 1*absmags[i],
                                       dx=0.001*absmags[i], order=5)
            assert abs(absmags_grad3/absmags_grad2[i] - 1)\
                < relative_accuracy

        np.testing.assert_allclose(colors_grad1, colors_grad2,
                                   rtol=relative_accuracy)
        for i in range(nobj):
            for j in range(ncols):
                def f(d):
                    colors2 = 1*colors
                    colors2[i, j] = d
                    return lnprob_nomarg(
                        nobj, nbins, ncols, varpi, varpi_err,
                        obsmags, obsmags_err, obscolors, obscolors_err,
                        absmags, distances, colors2, binamps, binmus, binsigs)
                colors_grad3 = derivative(f, 1*colors[i, j],
                                          dx=0.001*colors[i, j], order=5)
                assert abs(colors_grad3/colors_grad2[i, j] - 1)\
                    < relative_accuracy

        np.testing.assert_allclose(binamps_grad1, binamps_grad2,
                                   rtol=relative_accuracy)
        for b in range(nbins):
            def f(d):
                binamps2 = 1*binamps
                binamps2[b] = d
                return lnprob_nomarg(
                    nobj, nbins, ncols, varpi, varpi_err,
                    obsmags, obsmags_err, obscolors, obscolors_err,
                    absmags, distances, colors, binamps2, binmus, binsigs)
            binamps_grad3 = derivative(f, 1*binamps[b],
                                       dx=0.001*binamps[b], order=5)
            assert abs(binamps_grad3/binamps_grad2[b] - 1)\
                < relative_accuracy


def myprob_distgrid_marg(
    distances_grid,
    nobj, nbins, ncols,
    varpi, varpi_err,  # nobj
    obsmags, obsmags_err,  # nobj
    obscolors, obscolors_err,  # nobj, ncols
    bins,  # nobj
    binamps,  # nbins
    binmus,  # nbins, ncols + 1
    binsigs  # nbins, ncols + 1
        ):

    nbinsdist = distances_grid.size
    probgrid = np.zeros((nobj, nbinsdist))
    probgrid[:, :] = gaussian(1/distances_grid[None, :],
                              varpi[:, None], varpi_err[:, None])

    allbinmus = np.zeros((nobj, ncols + 1))
    allbinsigs = np.zeros((nobj, ncols + 1))
    famps = np.zeros((nobj, ))
    for o in range(nobj):
        famps[o] = binamps[bins[o]]
        allbinmus[o, 0] = obsmags[o] - binmus[bins[o], 0]
        allbinsigs[o, 0] = np.sqrt(binsigs[bins[o], 0]**2 +
                                   obsmags_err[o]**2)
        for i in range(ncols):
            allbinmus[o, i + 1] = binmus[bins[o], i + 1]
            allbinsigs[o, i + 1] = np.sqrt(binsigs[bins[o], i + 1]**2 +
                                           obscolors_err[o, i]**2)

    probgrid[:, :] *= famps[:, None]\
        * gaussian(5*np.log10(distances_grid)[None, :] + 10,
                   allbinmus[:, 0, None], allbinsigs[:, 0, None])
    for i in range(ncols):
        probgrid[:, :] *= gaussian(obscolors[:, None, i],
                                   allbinmus[:, i+1, None],
                                   allbinsigs[:, i+1, None])

    return probgrid


def myprob_bingrid_marg(
    nobj, nbins, ncols,
    varpi, varpi_err,  # nobj
    obsmags, obsmags_err,  # nobj
    obscolors, obscolors_err,  # nobj, ncols
    distances,  # nobj
    binamps,  # nbins
    binmus,  # nbins, ncols + 1
    binsigs  # nbins, ncols + 1
        ):

    probgrid = np.zeros((nobj, nbins))
    probgrid[:, :] = gaussian(1/distances, varpi, varpi_err)[:, None]

    sig = np.sqrt(binsigs[None, :, 0]**2 + obsmags_err[:, None]**2)
    probgrid[:, :] *= binamps[None, :]\
        * gaussian(5*np.log10(distances)[:, None] + 10,
                   obsmags[:, None] - binmus[None, :, 0], sig)
    for i in range(ncols):
        sig = np.sqrt(binsigs[None, :, i+1]**2 + obscolors_err[:, i, None]**2)
        probgrid[:, :] *= gaussian(obscolors[:, None, i],
                                   binmus[None, :, i+1], sig)

    return probgrid


def mylnprob_and_grads_marg(
    nobj, nbins, ncols,
    varpi, varpi_err,  # nobj
    obsmags, obsmags_err,  # nobj
    obscolors, obscolors_err,  # nobj, ncols
    bins,  # nobj
    distances,  # nobj
    binamps,  # nbins
    binmus,  # nbins, ncols + 1
    binsigs  # nbins, ncols + 1
        ):

    lnprobval = np.sum(lngaussian(1/distances, varpi, varpi_err))

    allbinmus = np.zeros((nobj, ncols + 1))
    allbinsigs = np.zeros((nobj, ncols + 1))
    famps = np.zeros((nobj, ))
    for o in range(nobj):
        famps[o] = binamps[bins[o]]
        allbinmus[o, 0] = obsmags[o] - binmus[bins[o], 0]
        allbinsigs[o, 0] = np.sqrt(binsigs[bins[o], 0]**2 +
                                   obsmags_err[o]**2)
        for i in range(ncols):
            allbinmus[o, i + 1] = binmus[bins[o], i + 1]
            allbinsigs[o, i + 1] = np.sqrt(binsigs[bins[o], i + 1]**2 +
                                           obscolors_err[o, i]**2)
    lnprobval += -np.sum(np.log(famps))
    lnprobval += np.sum(lngaussian(
            5*np.log10(distances) + 10,
            allbinmus[:, 0], allbinsigs[:, 0]))
    lnprobval += np.sum(lngaussian(
            obscolors, allbinmus[:, 1:], allbinsigs[:, 1:]))

    distances_grad = - lngaussian_grad(1/distances, varpi,
                                       varpi_err) / distances**2
    distances_grad += (5*np.log10(distances) + 10 - allbinmus[:, 0]) *\
        5.0 / (allbinsigs[:, 0]**2 * distances * np.log(10))

    return lnprobval, distances_grad


def test_SimpleHDRModel_marg_gradients():

    for k in range(NREPEAT):

        nbins = np.random.randint(4, 100)
        nobj = np.random.randint(10, 100)
        ncols = np.random.randint(1, 3)

        absmags = np.random.uniform(1, 2, nobj)
        distances = np.random.uniform(0.1, 0.3, nobj)
        varpi = 1/distances
        varpi_err = varpi*0.1
        varpi += varpi_err*np.random.randn(*varpi.shape)
        colors = np.random.uniform(1, 2, nobj*ncols).reshape((nobj, ncols))
        binamps = np.random.uniform(0, 1, nbins)
        binmus = np.random.uniform(1, 2, nbins*(ncols+1))\
            .reshape((nbins, ncols+1))
        bins = np.random.randint(low=0, high=nbins-1, size=nobj)
        binsigs = np.repeat(0.5, nbins*(ncols+1)).reshape((nbins, ncols+1))
        obsmags = absmags + 5*np.log10(distances) + 10
        obsmags_err = obsmags*0.1
        obsmags += obsmags_err * np.random.randn(*obsmags.shape)
        obscolors = 1*colors
        obscolors_err = obscolors*0.1
        obscolors += obscolors_err*np.random.randn(*colors.shape)

        lnprobval2, distances_grad2 =\
            mylnprob_and_grads_marg(
                nobj, nbins, ncols, varpi, varpi_err,
                obsmags, obsmags_err, obscolors, obscolors_err,
                bins, distances, binamps, binmus, binsigs)

        assert distances_grad2.size == nobj

        lnprobval1 = lnprob_marg(
            nobj, nbins, ncols, varpi, varpi_err,
            obsmags, obsmags_err, obscolors, obscolors_err,
            bins, distances, binamps, binmus, binsigs)

        assert (np.abs(lnprobval2/lnprobval1) - 1) < relative_accuracy

        distances_grad1 = 0*distances_grad2
        lnprob_distgradient_marg(
            distances_grad1,
            nobj, nbins, ncols, varpi, varpi_err,
            obsmags, obsmags_err, obscolors, obscolors_err,
            bins, distances, binamps, binmus, binsigs)
        np.testing.assert_allclose(distances_grad1, distances_grad2,
                                   rtol=relative_accuracy)
        for i in range(nobj):
            def f(d):
                distances2 = 1*distances
                distances2[i] = d
                lnprobval3 = lnprob_marg(
                    nobj, nbins, ncols, varpi, varpi_err,
                    obsmags, obsmags_err, obscolors, obscolors_err,
                    bins, distances2, binamps, binmus, binsigs)
                return lnprobval3

            distances_grad3 = derivative(f, 1*distances[i],
                                         dx=0.001*distances[i], order=5)
            assert abs(distances_grad3/distances_grad2[i] - 1) \
                < relative_accuracy

    probgrid1 = myprob_bingrid_marg(
        nobj, nbins, ncols, varpi, varpi_err,
        obsmags, obsmags_err, obscolors, obscolors_err,
        distances, binamps, binmus, binsigs)
    probgrid2 = 0*probgrid1
    prob_bingrid_marg(
        probgrid2,
        nobj, nbins, ncols, varpi, varpi_err,
        obsmags, obsmags_err, obscolors, obscolors_err,
        distances, binamps, binmus, binsigs)

    np.testing.assert_allclose(probgrid1, probgrid2,
                               rtol=relative_accuracy)

    distances_grid = np.linspace(0.2, 0.3, 4)
    probgrid1 = myprob_distgrid_marg(
        distances_grid,
        nobj, nbins, ncols, varpi, varpi_err,
        obsmags, obsmags_err, obscolors, obscolors_err,
        bins, binamps, binmus, binsigs)
    probgrid2 = 0*probgrid1
    prob_distgrid_marg(
        probgrid2, distances_grid.size, distances_grid,
        nobj, nbins, ncols, varpi, varpi_err,
        obsmags, obsmags_err, obscolors, obscolors_err,
        bins, binamps, binmus, binsigs)

    np.testing.assert_allclose(probgrid1, probgrid2,
                               rtol=relative_accuracy)

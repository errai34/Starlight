
import numpy as np
from scipy.misc import derivative

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
    binprobs = binamps[None, :] / nbins * gaussian(
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

    oldterm = binamps[None, :] / nbins * gaussian(
            absmags[:, None], binmus[None, :, 0], binsigs[None, :, 0])
    newterm = binamps[None, :] / nbins * gaussian_grad(
            absmags[:, None], binmus[None, :, 0], binsigs[None, :, 0])
    absmags_grad =\
        - (binprobs * newterm / oldterm).sum(axis=1) / binprobs.sum(axis=1)
    colors_grad = np.zeros((nobj, ncols))
    for i in range(ncols):
        colors_grad[:, i] = lngaussian_grad(colors[:, i],
                                            obscolors[:, i],
                                            obscolors_err[:, i])
        oldterm = binamps[None, :] / nbins * gaussian(
                colors[:, i, None],
                binmus[None, :, i + 1],
                binsigs[None, :, i + 1])
        newterm = binamps[None, :] / nbins * gaussian_grad(
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


def mylnprob_and_grads_marg(
    nobj, nbins, ncols,
    varpi, varpi_err,  # nobj
    obsmags, obsmags_err,  # nobj
    obscolors, obscolors_err,  # nobj, ncols
    distances,  # nobj
    binamps,  # nbins
    binmus,  # nbins, ncols + 1
    binsigs  # nbins, ncols + 1
        ):

    lnprobval = np.sum(
        lngaussian(1/distances, varpi, varpi_err)
        )
    sigb0 = np.sqrt(binsigs[None, :, 0]**2 + obsmags_err[:, None]**2)
    binprobs = binamps[None, :] / nbins * gaussian(
            5*np.log10(distances[:, None]) + 10,
            obsmags[:, None] + binmus[None, :, 0],
            sigb0
            )
    for i in range(ncols):
        sigbcol = np.sqrt(binsigs[None, :, i + 1]**2 +
                          obscolors_err[:, None, i]**2)
        binprobs *= gaussian(
                obscolors[:, None, i],
                binmus[None, :, i + 1],
                sigbcol
                )
    lnprobval += - np.log(np.sum(binprobs, axis=1)).sum()

    binamps_grad = - np.sum(binprobs / binamps[None, :] /
                            np.sum(binprobs, axis=1)[:, None], axis=0)

    distances_grad = - (1/distances - varpi) / (varpi_err * distances)**2
    gradterm = (5*np.log10(distances)[:, None] + 10 -
                obsmags[:, None] - binmus[None, :, 0]) *\
        5.0 / (sigb0**2 * distances[:, None] * np.log(10))
    distances_grad += np.sum(binprobs*gradterm, axis=1) /\
        np.sum(binprobs, axis=1)

    return lnprobval, distances_grad, binamps_grad


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

        lnprobval2, distances_grad2, binamps_grad2 =\
            mylnprob_and_grads_marg(
                nobj, nbins, ncols, varpi, varpi_err,
                obsmags, obsmags_err, obscolors, obscolors_err,
                distances, binamps, binmus, binsigs)

        assert binamps_grad2.size == nbins
        assert distances_grad2.size == nobj

        lnprobval1 = lnprob_marg(
            nobj, nbins, ncols, varpi, varpi_err,
            obsmags, obsmags_err, obscolors, obscolors_err,
            distances, binamps, binmus, binsigs)

        assert (np.abs(lnprobval2/lnprobval1) - 1) < relative_accuracy

        distances_grad1, binamps_grad1 = 0*distances_grad2, 0*binamps_grad2

        lnprob_gradients_marg(
            distances_grad1, binamps_grad1,
            nobj, nbins, ncols, varpi, varpi_err,
            obsmags, obsmags_err, obscolors, obscolors_err,
            distances, binamps, binmus, binsigs)

        np.testing.assert_allclose(binamps_grad1, binamps_grad2,
                                   rtol=relative_accuracy)
        for b in range(nbins):
            def f(d):
                binamps2 = 1*binamps
                binamps2[b] = d
                lnprobval3, distances_grad3, binamps_grad3 =\
                    mylnprob_and_grads_marg(
                        nobj, nbins, ncols, varpi, varpi_err,
                        obsmags, obsmags_err, obscolors, obscolors_err,
                        distances, binamps2, binmus, binsigs)
                return lnprobval3
            binamps_grad3 = derivative(f, 1*binamps[b],
                                       dx=0.001*binamps[b], order=5)
            assert abs(binamps_grad3/binamps_grad2[b] - 1)\
                < relative_accuracy

        np.testing.assert_allclose(distances_grad1, distances_grad2,
                                   rtol=relative_accuracy)
        for i in range(nobj):
            def f(d):
                distances2 = 1*distances
                distances2[i] = d
                lnprobval3, distances_grad3, binamps_grad3 =\
                    mylnprob_and_grads_marg(
                        nobj, nbins, ncols, varpi, varpi_err,
                        obsmags, obsmags_err, obscolors, obscolors_err,
                        distances2, binamps, binmus, binsigs)
                return lnprobval3

            distances_grad3 = derivative(f, 1*distances[i],
                                         dx=0.001*distances[i], order=5)
            assert abs(distances_grad3/distances_grad2[i] - 1) \
                < relative_accuracy

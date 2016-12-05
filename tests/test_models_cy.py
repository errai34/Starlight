
import numpy as np
from scipy.misc import derivative

from starlight.models_cy import *

relative_accuracy = 0.01
NREPEAT = 10


def gaussian(x, mu, sig):
    return np.exp(-0.5*((x - mu)/sig)**2) / np.sqrt(2*np.pi) / sig


def gaussian_grad(x, mu, sig):
    return np.exp(-0.5*((x - mu)/sig)**2) / np.sqrt(2*np.pi) / sig\
        * - ((x - mu)/sig**2)


def lngaussian(x, mu, sig):
    return 0.5*((x-mu)/sig)**2 + 0.5*np.log(2*np.pi) + np.log(sig)


def lngaussian_grad(x, mu, sig):
    return (x-mu)/sig**2


def allclose(v1, v2):
    #print(v1/v2-1)
    #print(np.max(np.abs(v1/v2)-1))
    #assert np.max(np.abs(v1/v2) - 1) < relative_accuracy
    #  print(np.max(np.abs(v1/v2)-1))
    np.testing.assert_allclose(v1, v2, rtol=relative_accuracy)
    #  assert np.all((v1/v2 - 1) < relative_accuracy)

def mylnprob_and_grads(
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
        binprobs *= binamps[None, :] / nbins * gaussian(
                colors[:, i, None], binmus[None, :, i + 1], binsigs[None, :, i + 1])
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
        colors_grad[:, i] = lngaussian_grad(colors[:, i], obscolors[:, i], obscolors_err[:, i])
        oldterm = binamps[None, :] / nbins * gaussian(
                colors[:, i, None], binmus[None, :, i + 1], binsigs[None, :, i + 1])
        newterm = binamps[None, :] / nbins * gaussian_grad(
                colors[:, i, None], binmus[None, :, i + 1], binsigs[None, :, i + 1])
        colors_grad[:, i] +=\
            - (binprobs * newterm / oldterm).sum(axis=1) / binprobs.sum(axis=1)

    binamps_grad = - np.sum(
        binprobs[:, :] / binamps[None, :] / binprobs.sum(axis=1)[:, None],
        axis=0)

    absmags_grad += (absmags+5*np.log10(distances)+10 - obsmags) / obsmags_err**2
    distances_grad = 5 * (absmags+5*np.log10(distances)+10 - obsmags) /\
        obsmags_err**2 / distances
    distances_grad += - (1/distances - varpi) / (varpi_err * distances)**2

    return lnprobval, absmags_grad, distances_grad, colors_grad, binamps_grad


def test_SimpleHDRModel_gradients():

    nbins = 4
    nobj = 4
    ncols = 2

    absmags = np.random.uniform(1, 2, nobj)
    distances = np.random.uniform(0.1, 0.3, nobj)
    varpi = 1/distances
    varpi_err = varpi*0.01
    varpi += varpi_err*np.random.randn(*varpi.shape)
    colors = np.random.uniform(1, 2, nobj*ncols).reshape((nobj, ncols))
    binamps = np.random.uniform(0, 1, nbins)
    binmus = np.random.uniform(1, 2, nbins*(ncols+1)).reshape((nbins, ncols+1))
    binsigs = np.repeat(0.5, nbins*(ncols+1)).reshape((nbins, ncols+1))
    obsmags = absmags + 5*np.log10(distances) + 10
    obsmags_err = obsmags*0.01
    obsmags += obsmags_err * np.random.randn(*obsmags.shape)
    obscolors = 1*colors
    obscolors_err = obscolors*0.01
    obscolors += obscolors_err*np.random.randn(*colors.shape)

    lnprobval2, absmags_grad2, distances_grad2, colors_grad2, binamps_grad2 =\
        mylnprob_and_grads(
            nobj, nbins, ncols, varpi, varpi_err,
            obsmags, obsmags_err, obscolors, obscolors_err,
            absmags, distances, colors, binamps, binmus, binsigs)

    lnprobval1 = lnprob(
        nobj, nbins, ncols, varpi, varpi_err,
        obsmags, obsmags_err, obscolors, obscolors_err,
        absmags, distances, colors, binamps, binmus, binsigs)

    assert (np.abs(lnprobval2/lnprobval1) - 1) < relative_accuracy

    absmags_grad1, distances_grad1, colors_grad1, binamps_grad1 =\
        0*absmags_grad2, 0*distances_grad2, 0*colors_grad2, 0*binamps_grad2

    lnprob_gradients(
        absmags_grad1, distances_grad1, colors_grad1, binamps_grad1,
        nobj, nbins, ncols, varpi, varpi_err,
        obsmags, obsmags_err, obscolors, obscolors_err,
        absmags, distances, colors, binamps, binmus, binsigs)

    allclose(distances_grad1, distances_grad2)
    allclose(absmags_grad1, absmags_grad2)
    allclose(colors_grad1, colors_grad2)
    allclose(binamps_grad1, binamps_grad2)

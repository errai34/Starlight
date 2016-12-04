
import numpy as np


class SimpleGaussianModel:

    def __init__(self):
        pass

    def strip_params(self, x):
        #  params: mu, sig
        assert x.size == 2
        return x[0], x[1]

    def combine_params(self, mu, sig):
        return np.array([mu, sig])

    def draw(self, x, nobj):
        mu, sig = self.strip_params(x)
        self.samples = mu + sig * np.random.randn(nobj)
        self.nobj = nobj

    def log_posterior(self, x):
        mu, sig = self.strip_params(x)
        return - np.sum(- 0.5 * ((self.samples - mu)/sig)**2 -
                      np.log(np.pi) - np.log(sig))

    def log_posterior_gradients(self, x):
        mu, sig = self.strip_params(x)
        mu_grad = - np.sum((self.samples - mu)/sig**2)
        sig_grad = - np.sum((self.samples - mu)**2/sig**3 - 1/sig)
        return self.combine_params(mu_grad, sig_grad)


class FlatModel:

    def __init__(self, varpi, varpi_err, mags, mags_err,
                 binmus, binsigs):
        self.nobj, self.ndim = mags.shape
        assert mags.shape == mags_err.shape
        assert varpi.size == self.nobj
        assert varpi_err.size == self.nobj
        self.nbins = binmus.shape[0]
        assert binsigs.shape[0] == self.nbins
        assert binmus.shape[1] == self.ndim
        assert binsigs.shape[1] == self.ndim

        self.nparams = self.nbins + self.nobj * (self.ndim + 1)
        self.varpi, self.varpi_err = varpi, varpi_err
        self.mags, self.mags_err = mags, mags_err
        self.binmus, self.binsigs = binmus, binsigs
        self.splits = [self.nobj,
                       self.nobj * 2,
                       self.nobj * (2 + self.ndim)]

    def log_posterior(self, x):
        absmag, distances, colors, binamps = np.split(x, self.splits)
        return lnprob(self.nobj, self.nbins, self.ndim,
                      self.varpi, self.varpi_err, self.mags, self.mags_err,
                      absmag, distances, colors, binamps,
                      self.binmus, self.binsigs)

    def log_posterior_gradients(self, x):
        absmag, distances, colors, binamps = np.split(x, self.splits)
        absmag_grad, distances_grad, colors_grad, binamps_grad =\
            0*absmag, 0*distances, 0*colors, 0*binamps
        lnprob_gradients(
            self.nobj, self.nbins, self.ndim,
            absmag_grad, distances_grad, colors_grad, binamps_grad,
            self.varpi, self.varpi_err, self.mags, self.mags_err,
            absmag, distances, colors, binamps,
            self.binmus, self.binsigs)
        return np.concatenate((absmag_grad, distances_grad,
                               colors_grad, binamps_grad))

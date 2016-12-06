
import numpy as np

from starlight.hmc import hmc_sampler
from starlight.models import SimpleGaussianModel, SimpleHRDModel

NREPEAT = 1


def test_HMC_SimpleGaussianModel():

    for i in range(NREPEAT):
        mu, sig = np.random.uniform(low=0.5, high=1, size=2)
        nobj = 1000
        model = SimpleGaussianModel()
        x_true = model.combine_params(mu, sig)
        model.draw(x_true, nobj)

        num_samples = 1000
        param_samples = np.zeros((num_samples, 2))
        param_samples[0, :] = np.random.uniform(low=0.2, high=1, size=2)
        for i in range(1, num_samples):
            param_samples[i, :] = hmc_sampler(param_samples[i-1, :],
                                              model, step_size=1e-3,
                                              num_steps=40)
        x_inf_mean = param_samples.mean(axis=0)
        x_inf_std = param_samples.std(axis=0)
        chi2 = ((x_true - x_inf_mean)/x_inf_std)**2
        print(mu, sig)
        print(x_inf_mean)
        print(x_inf_std)
        print(chi2)
        assert np.all(chi2 < 5)


def test_HMC_SimpleHRDModel():

    for k in range(NREPEAT):
        nbins_perdim = np.random.randint(2, 6)
        ncols = np.random.randint(1, 3)
        nobj = np.random.randint(20, 50)
        varpi_fracerror, mags_fracerror = np.random.uniform(0.001, 0.01, 2)

        model = SimpleHRDModel()
        nbins, binamps, binmus, binsigs = model.draw_bins(nbins_perdim, ncols)
        absmags, colors, distances,\
            varpi, varpi_err,\
            obsmags, obsmags_err,\
            obscolors, obscolors_err\
            = model.draw(binamps, binmus, binsigs,
                         varpi_fracerror, mags_fracerror, nobj)

        model.set_data(binmus, binsigs, varpi, varpi_err,
                       obsmags, obsmags_err, obscolors, obscolors_err)
        x_true = model.combine_params(distances, binamps)

        num_samples = 10
        param_samples = np.zeros((num_samples, x_true.size))
        param_samples[0, :] = x_true
        for i in range(1, num_samples):
            param_samples[i, :] = hmc_sampler(param_samples[i-1, :],
                                              model, step_size=1e-4,
                                              num_steps=10)
        x_inf_mean = param_samples.mean(axis=0)
        x_inf_std = param_samples.std(axis=0)
        chi2 = ((x_true - x_inf_mean)/x_inf_std)**2
        print(x_true)
        print(x_inf_mean)
        print(x_inf_std)
        print(chi2)
        assert np.all(chi2 < 100)

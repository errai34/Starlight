
import numpy as np

from starlight.hmc import hmc_sampler
from starlight.models import SimpleGaussianModel, SimpleHRDModel
from starlight.models_cy import lnprob_distgradient_marg

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
        assert np.all(chi2 < 15)

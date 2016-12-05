
import numpy as np
from scipy.misc import derivative

from starlight.models import *

relative_accuracy = 0.01
NREPEAT = 1


def test_SimpleGaussianModel_gradients():

    for k in range(NREPEAT):
        mu, sig = np.random.uniform(low=0.2, high=1, size=2)
        nobj = 100
        model = SimpleGaussianModel()
        x = model.combine_params(mu, sig)
        model.draw(x, nobj)

        mu_grad, sig_grad =\
            model.strip_params(model.log_posterior_gradients(x))

        def f(mu_bis):
            x = model.combine_params(mu_bis, sig)
            return model.log_posterior(x)

        mu_grad2 = derivative(f, mu, dx=0.001*mu)
        assert abs(mu_grad2/mu_grad-1) < relative_accuracy

        def f(sig_bis):
            x = model.combine_params(mu, sig_bis)
            return model.log_posterior(x)

        sig_grad2 = derivative(f, sig, dx=0.001*sig)
        assert abs(sig_grad2/sig_grad-1) < relative_accuracy


    # ADD OPTIMIZATION

def test_SimpleHDRModel_gradients():

    nbins_perdim = 2
    ncols = 1
    nobj = 10
    varpi_fracerror, mags_fracerror = 0.001, 0.001

    for k in range(NREPEAT):
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

        x = model.combine_params(absmags, distances, colors, binamps)
        absmag_grad, distances_grad, colors_grad, binamps_grad =\
            model.strip_params(model.log_posterior_gradients(x))
        print("LNPOST", model.log_posterior(x))

        def f(distances2):
            x = model.combine_params(absmags, distances2, colors, binamps)
            return model.log_posterior(x)

        distances_grad2 = derivative(f, distances, dx=0.001*distances)
        np.testing.assert_allclose(distances_grad2, distances_grad, rtol=relative_accuracy)

    # ADD OPTIMIZATION

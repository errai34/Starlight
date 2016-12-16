
import numpy as np
from scipy.misc import derivative
import pytest
from starlight.models import *
from scipy.optimize import minimize
from starlight.models_cy import lnprob_distgradient_marg

relative_accuracy = 0.02
NREPEAT = 2


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


@pytest.mark.skip(reason="Annoying tests")
def test_SimpleHDRModel_nomarg_gradients():

    for k in range(NREPEAT):
        nbins_perdim = np.random.randint(10, 60)
        ncols = np.random.randint(1, 3)
        nobj = np.random.randint(10, 100)
        varpi_fracerror, mags_fracerror = np.random.uniform(0.01, 0.02, 2)

        model = SimpleHRDModel_nomarg()
        nbins, binamps, binmus, binsigs = model.draw_bins(nbins_perdim, ncols)
        absmags, colors, distances =\
            model.draw_properties(binamps, binmus, binsigs, nobj)
        varpi, varpi_err, obsmags, obsmags_err, obscolors, obscolors_err =\
            model.draw_data(absmags, colors, distances,
                            varpi_fracerror, mags_fracerror)

        model.set_data(binmus, binsigs, varpi, varpi_err,
                       obsmags, obsmags_err, obscolors, obscolors_err)

        x = model.combine_params(absmags, distances, colors, binamps)
        absmag_grad, distances_grad, colors_grad, binamps_grad =\
            model.strip_params(model.log_posterior_gradients(x))

        for i in range(nbins):

            def f(d):
                binamps2 = 1*binamps
                binamps2[i] = d
                x = model.combine_params(absmags, distances,
                                         colors, binamps2)
                return model.log_posterior(x)

            binamps_grad2 = derivative(f, binamps[i],
                                       dx=0.001*binamps[i], order=7)
            np.testing.assert_allclose(binamps_grad2,
                                       binamps_grad[i],
                                       rtol=relative_accuracy)

        for i in range(nobj):

            def f(d):
                absmags2 = 1*absmags
                absmags2[i] = d
                x = model.combine_params(absmags2, distances,
                                         colors, binamps)
                return model.log_posterior(x)

            absmag_grad2 = derivative(f, absmags[i],
                                      dx=0.001*absmags[i], order=5)
            np.testing.assert_allclose(absmag_grad2,
                                       absmag_grad[i],
                                       rtol=relative_accuracy)

            def f(d):
                distances2 = 1*distances
                distances2[i] = d
                x = model.combine_params(absmags, distances2,
                                         colors, binamps)
                return model.log_posterior(x)

            distances_grad2 = derivative(f, distances[i],
                                         dx=0.001*distances[i], order=5)
            np.testing.assert_allclose(distances_grad2,
                                       distances_grad[i],
                                       rtol=relative_accuracy)

            for j in range(ncols):
                def f(d):
                    colors2 = 1*colors
                    colors2[i, j] = d
                    x = model.combine_params(absmags, distances,
                                             colors2, binamps)
                    return model.log_posterior(x)

                colors_grad2 = derivative(f, colors[i, j],
                                          dx=0.001*colors[i, j], order=5)
                np.testing.assert_allclose(colors_grad2,
                                           colors_grad[i, j],
                                           rtol=relative_accuracy)


def test_SimpleHDRModel_gradients():

    for k in range(NREPEAT):
        nbins_perdim = np.random.randint(10, 20)
        ncols = np.random.randint(1, 2)
        nobj = np.random.randint(10, 20)
        varpi_fracerror, mags_fracerror = np.random.uniform(0.0001, 0.02, 2)

        model = SimpleHRDModel()
        nbins, binamps, binmus, binsigs = model.draw_bins(nbins_perdim, ncols)
        absmags, colors, distances, bins =\
            model.draw_properties(binamps, binmus, binsigs, nobj)
        varpi, varpi_err, obsmags, obsmags_err, obscolors, obscolors_err =\
            model.draw_data(absmags, colors, distances,
                            varpi_fracerror, mags_fracerror)

        model.set_data(binmus, binsigs, varpi, varpi_err,
                       obsmags, obsmags_err, obscolors, obscolors_err)

        distances_samples, bins_samples, binamps_samples =\
            model.gibbs_sampler(1, num_steps=10)

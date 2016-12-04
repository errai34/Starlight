
import numpy as np
from scipy.misc import derivative

from starlight.models import SimpleGaussianModel

relative_accuracy = 0.01
NREPEAT = 10


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

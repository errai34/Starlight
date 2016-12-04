
import numpy as np


def hmc_sampler(x0, model, step_size, num_steps, mass_matrix=None):
    if mass_matrix is None:
        inv_mass_matrix = np.eye(x0.size)
    else:
        assert mass_matrix.size == x0.size**2
        inv_mass_matrix = np.linalg.inv(mass_matrix)
    v0 = np.random.randn(x0.size)
    v = v0 - 0.5 * step_size * model.log_posterior_gradients(x0)
    x = x0 + step_size * v
    for i in range(num_steps):
        v = v - step_size * model.log_posterior_gradients(x)
        x = x + step_size * v
    v = v - 0.5 * step_size * model.log_posterior_gradients(x)
    orig = model.log_posterior(x0)\
        + 0.5 * np.dot(v0.T, np.dot(inv_mass_matrix, v0))
    current = model.log_posterior(x)\
        + 0.5 * np.dot(v.T, np.dot(inv_mass_matrix, v))
    p_accept = min(1.0, np.exp(orig - current))
    if p_accept > np.random.uniform():
        return x
    else:
        return x0

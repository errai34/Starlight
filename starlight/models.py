
import numpy as np
from scipy.optimize import minimize

from starlight.models_cy import lnprob_distgradient_marg,\
    lnprob_marg, prob_bingrid_fullmarg, sample_bins_from_grid


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


class SimpleHRDModel_nomarg:

    def __init__(self):
        pass

    def draw_bins(self, nbins_perdim, ncols):
        grids = np.meshgrid(*[np.linspace(0, 1, nbins_perdim)
                              for i in range(ncols + 1)])
        binmus = np.vstack([g.ravel() for g in grids]).T
        binsigs = 0*binmus + 1./nbins_perdim
        nbins = nbins_perdim**(ncols + 1)
        binamps = np.random.dirichlet(np.repeat(1, nbins))
        return nbins, binamps, binmus, binsigs

    def strip_params(self, x):
        assert x.size == self.nobj * (2 + self.ncols) + self.nbins
        absmags, distances, colors, binamps = np.split(x, self.splits)
        colors = colors.reshape((self.nobj, self.ncols))
        return absmags, distances, colors, binamps

    def combine_params(self, absmags, distances, colors, binamps):
        return np.concatenate([absmags, distances, colors.ravel(), binamps])

    def draw_properties(self, binamps, binmus, binsigs, nobj):
        self.nobj = nobj
        self.nbins = binamps.shape[0]
        self.ncols = binmus.shape[1] - 1
        assert binmus.shape[0] == self.nbins
        assert binsigs.shape[0] == self.nbins
        assert binsigs.shape[1] == binmus.shape[1]
        self.splits = [self.nobj,
                       self.nobj * 2,
                       self.nobj * (2 + self.ncols)]
        # draw bins from bin amps
        bincounts = np.random.multinomial(nobj, binamps)
        self.bincounts = bincounts
        self.bins = np.zeros((nobj, ), dtype=int)
        cumcounts = np.concatenate((np.array([0]), np.cumsum(bincounts)))
        # draw absmags and colors from gaussians
        absmags, colors = np.zeros((nobj, )), np.zeros((nobj, self.ncols))
        for b in range(self.nbins):
            start, end = cumcounts[b], cumcounts[b+1]
            self.bins[start:end] = b
            absmags[start:end] = binmus[b, 0] +\
                binsigs[b, 0] * np.random.randn(bincounts[b])
            for i in range(self.ncols):
                colors[start:end, i] = binmus[b, i + 1] +\
                    binsigs[b, i + 1] * np.random.randn(bincounts[b])
        # draw distances from uniform prior
        distances = np.random.uniform(low=0.1, high=0.3, size=nobj)
        return absmags, colors, distances, self.bins

    def draw_data(self, absmags, colors, distances,
                  varpi_fracerror, mags_fracerror):
        # draw parallaxes
        nobj = distances.size
        varpi = 1/distances
        varpi_err = varpi * varpi_fracerror
        varpi += varpi_err * np.random.randn(nobj)
        # draw apparent magnitudes and colors
        obsmags = absmags + 5*np.log10(distances) + 10
        obsmags_err = obsmags * mags_fracerror
        obsmags += obsmags_err * np.random.randn(nobj)
        obscolors = np.zeros((nobj, self.ncols))
        obscolors = 1*colors
        obscolors_err = obscolors*0
        for j in range(colors.shape[1]):
            obscolors_err[:, j] = obscolors[:, j] * mags_fracerror
            obscolors[:, j] = obscolors[:, j] + obscolors_err[:, j] *\
                np.random.randn(nobj)
        return varpi, varpi_err, obsmags, obsmags_err, obscolors, obscolors_err

    def set_data(self, binmus, binsigs, varpi, varpi_err,
                 obsmags, obsmags_err, obscolors, obscolors_err,
                 dist_min, dist_max):

        self.nobj, self.ncols = obscolors.shape
        self.nbins = binmus.shape[0]
        self.dist_min, self.dist_max = dist_min, dist_max
        assert binsigs.shape[0] == self.nbins
        assert binmus.shape[1] == self.ncols + 1
        assert binsigs.shape[1] == binmus.shape[1]
        assert varpi.size == self.nobj
        assert varpi_err.size == self.nobj
        self.varpi, self.varpi_err = varpi, varpi_err
        self.obsmags, self.obsmags_err = obsmags, obsmags_err
        self.obscolors, self.obscolors_err = obscolors, obscolors_err

        self.binmus, self.binsigs = binmus, binsigs

        self.splits = [self.nobj,
                       self.nobj * 2,
                       self.nobj * (2 + self.ncols)]

    def log_posterior(self, x):
        absmag, distances, colors, binamps = self.strip_params(x)
        return lnprob_nomarg(self.nobj, self.nbins, self.ncols,
                             self.varpi, self.varpi_err,
                             self.obsmags, self.obsmags_err,
                             self.obscolors, self.obscolors_err,
                             absmag, distances, colors, binamps,
                             self.binmus, self.binsigs)

    def log_posterior_gradients(self, x):
        absmag, distances, colors, binamps = self.strip_params(x)
        absmag_grad, distances_grad, colors_grad, binamps_grad =\
            0*absmag, 0*distances, 0*colors, 0*binamps
        lnprob_gradients_nomarg(
            absmag_grad, distances_grad, colors_grad, binamps_grad,
            self.nobj, self.nbins, self.ncols,
            self.varpi, self.varpi_err,
            self.obsmags, self.obsmags_err,
            self.obscolors, self.obscolors_err,
            absmag, distances, colors, binamps,
            self.binmus, self.binsigs)
        return self.combine_params(absmag_grad, distances_grad,
                                   colors_grad, binamps_grad)


class SimpleHRDModel(SimpleHRDModel_nomarg):

    def set_data(self, binmus, binsigs, varpi, varpi_err,
                 obsmags, obsmags_err, obscolors, obscolors_err,
                 dist_min, dist_max):

        super(SimpleHRDModel, self).set_data(binmus, binsigs,
                                             varpi, varpi_err,
                                             obsmags, obsmags_err,
                                             obscolors, obscolors_err,
                                             dist_min, dist_max)

        self.splits = [self.nobj]
        self.ibins = None
        self.probgrid_magsonly = None
        self.distances = None
        self.binamps = None
        self.bins = None
        self.binamps = None
        self.nearestbins = None
        self.counts = None
        self.bincounts = None
        self.allbinsigs = np.zeros((self.nobj, self.nbins, self.ncols + 1))
        self.allbinsigs[:, :, 0] = np.sqrt(binsigs[None, :, 0]**2 +
                                           obsmags_err[:, None]**2)
        for i in range(self.ncols):
            self.allbinsigs[:, :, i+1] = np.sqrt(binsigs[None, :, i+1]**2 +
                                                 obscolors_err[:, i, None]**2)

    def strip_params(self, x):
        assert x.size == self.nobj + self.nbins
        distances, binamps = np.split(x, self.splits)
        return distances, binamps

    def combine_params(self, distances, binamps):
        return np.concatenate([distances, binamps])

    def log_posterior(self, bins, distances, binamps):
        return lnprob_marg(self.nobj, self.nbins, self.ncols,
                           self.varpi, self.varpi_err,
                           self.obsmags, self.obsmags_err,
                           self.obscolors, self.obscolors_err,
                           bins, distances, binamps,
                           self.binmus, self.binsigs)

    def log_posterior_gradients(self, x):
        raise NotImplemented()

    def optimize(self, x_ini):
        raise NotImplemented()

    def mcmcdraw_bins(self):
        if self.ibins is None:
            self.ibins = np.repeat(np.arange(1, self.nbins), self.nobj)\
                .reshape((self.nbins-1, self.nobj)).T.ravel()
        if self.probgrid_magsonly is None:
            self.probgrid_magsonly = np.zeros((self.nobj, self.nbins))
            prob_bingrid_magsonly_marg(
                self.probgrid_magsonly, self.nobj, self.nbins, self.ncols,
                self.varpi, self.varpi_err, self.obsmags, self.obscolors,
                self.distances, self.binamps, self.binmus, self.allbinsigs)
        probgrid = 1*self.probgrid_magsonly
        prob_bingrid_distandbins_marg(
            probgrid, self.nobj, self.nbins, self.ncols,
            self.varpi, self.varpi_err, self.obsmags, self.obscolors,
            self.distances, self.binamps, self.binmus, self.allbinsigs)
        cumsumweights = np.add.accumulate(probgrid, axis=1).T
        cumsumweights /= cumsumweights[-1, :]
        pos = np.random.uniform(0.0, 1.0, size=self.nobj)
        cond = pos > cumsumweights[:-1, :]
        cond &= pos <= cumsumweights[1:, :]
        res = np.zeros(self.nobj, dtype=int)
        res[pos <= cumsumweights[0, :]] = 0
        locs = np.any(cond, axis=0)
        res[locs] = self.ibins[cond.T.ravel()]
        self.bins = res
        self.bincounts = np.bincount(res, minlength=self.nbins)
        return res

    def mcmcdraw_binamps(self):
        self.bincounts = np.bincount(self.bins, minlength=self.nbins)
        gammabs = np.array([np.random.gamma(alpha+1)
                            for alpha in self.bincounts])
        fbs = gammabs / gammabs.sum(axis=0)
        self.binamps = fbs
        return fbs

    def dist_bin_hessian(self, dist, bins):
        mus = self.binmus[bins, 0]
        sigs = self.binsigs[bins, 0]
        fac1 = - 2*(self.varpi - 1./dist)/self.varpi_err**2/dist**3
        fac1 += 1./dist**4/self.varpi_err**2
        fac2 = - (mus + 5*np.log10(dist) + 10 - self.obsmags) / \
            (sigs**2+self.obsmags_err**2) * 5/np.log(10) / dist**2
        fac2 += (5/np.log(10))**2 / (sigs**2+self.obsmags_err**2) / dist**2
        return fac1 + fac2

    def mcmcdraw_distances(self, num_steps=10, dist_min=0.0, dist_max=0.4,
                           step_size_min=1e-5, step_size_max=1e-2):
        accept = False
        naivedist = 1/self.varpi
        naivedist_err = self.varpi_err / self.varpi**2
        inv_mass_matrix_diag = 1./self.dist_bin_hessian(1/self.varpi, self.bins)

        if inv_mass_matrix_diag is None:
            inv_mass_matrix_diag_sqrt = np.repeat(1, self.nobj)
        else:
            assert inv_mass_matrix_diag.size == self.nobj
            inv_mass_matrix_diag_sqrt = inv_mass_matrix_diag**0.5

        while accept is False:  # TODO: improve that!

            step_size = step_size_min + (step_size_max - step_size_min) *\
                np.random.uniform(low=0, high=1, size=self.nobj)
            distances0 = 1*self.distances
            v0 = np.random.randn(distances0.size)

            distgrads = np.zeros((self.nobj, ))
            lnprob_distgradient_marg(
                distgrads, self.nobj, self.nbins, self.ncols,
                self.varpi, self.varpi_err, self.obsmags, self.obsmags_err,
                self.obscolors, self.obscolors_err,
                self.bins, distances0, self.binamps,
                self.binmus, self.binsigs)

            for j in range(12):
                v = v0 - 0.5 * step_size * distgrads
                distances = distances0 + step_size * v *\
                    inv_mass_matrix_diag_sqrt

                ind_upper = distances > dist_max
                distances[ind_upper] = 2*dist_max - distances[ind_upper]
                v[ind_upper] = - v[ind_upper]
                ind_lower = distances <= dist_min
                distances[ind_lower] = 2*dist_min - distances[ind_lower]
                v[ind_lower] = - v[ind_lower]
                ind_upper = distances > dist_max
                ind_lower = distances <= dist_min

                ind_bad = np.logical_or(ind_lower, ind_upper)
                ind_bad |= ((distances - naivedist)/naivedist_err)**2\
                    > 10
                if(ind_bad.sum() == 0):
                    break
                # print('Decreased stepsize for', ind_bad.sum(), 'objects')
                step_size[ind_bad] /= 10

            if np.sum(~np.isfinite(distgrads)) > 0\
                or np.sum(~np.isfinite(distances)) > 0\
                    or ind_lower.sum() > 0 or ind_upper.sum() > 0:
                print("num", np.sum(~np.isfinite(distgrads)),
                      np.sum(~np.isfinite(distances)),
                      ind_lower.sum(), ind_upper.sum())
                bad = ~np.isfinite(distances)
                bad |= ind_lower
                bad |= ind_upper
                print("dist0", distances0[bad])
                print("dist", distances[bad])
                print("disterr", (self.varpi_err/self.varpi**2)[bad])
                print("grad", distgrads[bad])
                print("step_sizes", step_size[bad])
                stop

            for i in range(num_steps):

                distgrads0 = 1*distgrads
                lnprob_distgradient_marg(
                    distgrads, self.nobj, self.nbins, self.ncols,
                    self.varpi, self.varpi_err, self.obsmags, self.obsmags_err,
                    self.obscolors, self.obscolors_err,
                    self.bins, distances, self.binamps,
                    self.binmus, self.binsigs)

                for j in range(12):
                    newv = v - step_size * distgrads
                    newdistances = distances + step_size * newv *\
                        inv_mass_matrix_diag_sqrt

                    ind_upper = newdistances > dist_max
                    newdistances[ind_upper] = 2*dist_max\
                        - newdistances[ind_upper]
                    newv[ind_upper] = - newv[ind_upper]
                    ind_lower = newdistances <= dist_min
                    newdistances[ind_lower] = 2*dist_min\
                        - newdistances[ind_lower]
                    newv[ind_lower] = - newv[ind_lower]
                    ind_upper = newdistances > dist_max
                    ind_lower = newdistances <= dist_min

                    ind_bad = np.logical_or(ind_bad, ind_bad)
                    ind_bad |= ((newdistances - naivedist)/naivedist_err)**2\
                        > 10
                    if(ind_bad.sum() == 0):
                        break
                    # print('Decreased stepsize (at leapfrog', i, ') for',
                    # ind_bad.sum(), 'objects')
                    step_size[ind_bad] /= 10

                distances = newdistances
                v = newv

            lnprob_distgradient_marg(
                distgrads, self.nobj, self.nbins, self.ncols,
                self.varpi, self.varpi_err, self.obsmags, self.obsmags_err,
                self.obscolors, self.obscolors_err,
                self.bins, distances, self.binamps,
                self.binmus, self.binsigs)
            v = v - 0.5 * step_size * distgrads

            lnprob = lnprob_marg(
                self.nobj, self.nbins, self.ncols,
                self.varpi, self.varpi_err,
                self.obsmags, self.obsmags_err,
                self.obscolors, self.obscolors_err,
                self.bins, distances, self.binamps,
                self.binmus, self.binsigs)
            lnprob0 = lnprob_marg(
                self.nobj, self.nbins, self.ncols,
                self.varpi, self.varpi_err,
                self.obsmags, self.obsmags_err,
                self.obscolors, self.obscolors_err,
                self.bins, distances0, self.binamps,
                self.binmus, self.binsigs)
            orig = lnprob0
            current = lnprob
            if inv_mass_matrix_diag is None:
                orig += 0.5 * np.dot(v0.T, v0)
                current += 0.5 * np.dot(v.T, v)
            else:
                orig += 0.5 * np.sum(inv_mass_matrix_diag * v0**2.)
                current += 0.5 * np.sum(inv_mass_matrix_diag * v**2.)

            if np.isfinite(orig) and np.isfinite(current):
                p_accept = min(1.0, np.exp(orig - current))
                accept = p_accept > np.random.uniform()
            if accept is False:
                print(lnprob0, np.dot(v0.T, v0))
                print(lnprob, np.dot(v.T, v))
                print("rejected", end=" - ")
                exit(1)

        # print('Final step_sizes', np.min(step_size),
        # np.mean(step_size), np.max(step_size))
        self.distances = distances
        ind_upper = distances > dist_max
        ind_lower = distances <= dist_min
        if np.sum(~np.isfinite(distgrads)) > 0\
            or np.sum(~np.isfinite(distances)) > 0\
                or ind_lower.sum() > 0 or ind_upper.sum() > 0:
            print("num const:", ind_lower.sum(), ind_upper.sum())
            print("num nan", np.sum(~np.isfinite(distgrads)),
                  np.sum(~np.isfinite(distances)))
            stop
        return distances

    def gibbs_sampler(self, num_samples, num_steps=10,
                      step_size_min=1e-6, step_size_max=1e-2, dist_max=1):
        self.probgrid = np.zeros((self.nobj, self.nbins))
        from time import time
        t1 = time()
        prob_bingrid_fullmarg(
            self.probgrid, self.dist_min, self.dist_max,
            self.nobj, self.nbins, self.ncols,
            self.varpi, self.varpi_err,
            self.obsmags, self.obsmags_err,
            self.obscolors, self.obscolors_err, self.binmus, self.binsigs)
        t2 = time()
        print("Precomputation took", t2-t1)
        self.bins = np.repeat(0, self.nobj).astype(int)
        self.binamps = np.repeat(1./self.nbins, self.nbins)

        bins_samples = np.zeros((num_samples, self.nobj))
        binamps_samples = np.zeros((num_samples, self.nbins))
        t1t, t2t = 0, 0
        for i in range(num_samples):
            t1 = time()
            sample_bins_from_grid(
                self.bins, self.probgrid,
                self.binamps, self.nobj, self.nbins)
            bins_samples[i, :] = self.bins
            t2 = time()
            binamps_samples[i, :] = self.mcmcdraw_binamps()
            t3 = time()
            t1t += (t2-t1)/num_samples
            t2t += (t3-t2)/num_samples

        print('Time per sample: %g' % t1t, 's , %g' % t2t, 's ')
        return bins_samples, binamps_samples

    def gibbs_sampler_withdist(self, num_samples, num_steps=10,
                               step_size_min=1e-6, step_size_max=1e-2,
                               dist_max=1):
        if self.distances is None:
            self.distances = 1./self.varpi
        if self.binamps is None or self.bins is None\
                or self.binamps is None or self.nearestbins is None\
                or self.counts is None or self.bincounts is None:
            self.binamps = np.repeat(1./self.nbins, self.nbins)
            self.bins = np.zeros((self.nobj, ), dtype=int)
            self.nearestbins = np.zeros((self.nobj, ), dtype=int)
            self.counts = np.zeros((self.nobj, ), dtype=int)
            sample_bins_marg(
                self.bins, self.nearestbins, self.counts,
                self.nobj, self.nbins, self.ncols,
                self.varpi, self.varpi_err, self.obsmags, self.obscolors,
                self.distances, self.binamps, self.binmus, self.allbinsigs)
            self.mcmcdraw_binamps()

        from time import time
        distances_samples = np.zeros((num_samples, self.nobj))
        bins_samples = np.zeros((num_samples, self.nobj))
        binamps_samples = np.zeros((num_samples, self.nbins))
        t1t, t2t, t3t = 0, 0, 0
        for i in range(num_samples):
            t1 = time()
            # bins_samples[i, :] = self.mcmcdraw_bins()
            sample_bins_marg(
                self.bins, self.nearestbins, self.counts,
                self.nobj, self.nbins, self.ncols,
                self.varpi, self.varpi_err, self.obsmags, self.obscolors,
                self.distances, self.binamps, self.binmus, self.allbinsigs)
            bins_samples[i, :] = self.bins
            t2 = time()
            distances_samples[i, :] = self.mcmcdraw_distances(
                num_steps=num_steps,
                step_size_min=step_size_min,
                step_size_max=step_size_max,
                dist_max=dist_max)
            t3 = time()
            binamps_samples[i, :] = self.mcmcdraw_binamps()
            t4 = time()
            t1t += (t2-t1)/num_samples
            t2t += (t3-t2)/num_samples
            t3t += (t4-t3)/num_samples

        print('Time per sample: %g' % t1t, 's , %g' % t2t, 's , %g' % t3t, 's')
        return distances_samples, bins_samples, binamps_samples

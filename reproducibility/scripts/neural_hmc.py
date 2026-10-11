# Our CPU-only archived research implementation; experimental, not default inference.
"""Neural-guided HMC proposals with exact physical-density acceptance.

The proposal potential may use the RF twice. The sampled target does not:
only the original physical density and one box Jacobian enter acceptance.
A fixed differentiable proposal potential generates reversible, volume-
preserving leapfrog maps; endpoint acceptance uses the true Hamiltonian.
Warmup step sizes/metrics freeze before production. No core GRTM edits.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

def guided_hmc_sample(physical, guide, initial, lower, upper, *, strength=0.1, steps=2200, burn_in=800, leapfrog_steps=8, step_size=0.05, target_accept=0.9, seed=0):
    x = torch.as_tensor(initial, dtype=torch.float64).clone()
    lo = torch.as_tensor(lower, dtype=x.dtype)
    hi = torch.as_tensor(upper, dtype=x.dtype)
    span = hi - lo
    if x.ndim != 2 or lo.shape != (x.shape[1],) or hi.shape != lo.shape:
        raise ValueError('Invalid state/bounds shape')
    if not bool(torch.isfinite(x).all() & (hi > lo).all() & (x > lo).all() & (x < hi).all()):
        raise ValueError('Require finite interior states and ordered bounds')
    if steps <= burn_in or burn_in < 150 or leapfrog_steps < 1 or (not np.isfinite(strength)) or (strength < 0):
        raise ValueError('Invalid sampling configuration')
    chains, dim = x.shape
    q = torch.logit((x - lo) / span)
    eye = torch.eye(dim, dtype=x.dtype).expand(chains, dim, dim).clone()
    metric = eye.clone()
    chol = metric.clone()
    rng = torch.Generator().manual_seed(seed)
    lengths = np.random.default_rng(seed).integers(max(1, leapfrog_steps // 2), 3 * leapfrog_steps // 2 + 1, steps)
    eps = torch.full((chains,), step_size, dtype=x.dtype)
    logbar = eps.log()
    mu = (10 * eps).log()
    hbar = torch.zeros_like(eps)
    dual = 0
    end = burn_in - 50
    point = 50
    width = 50
    updates = set()
    while point < end:
        point = end if end - point < 2 * width else point + width
        updates.add(point)
        width *= 2
    mean = q.clone()
    m2 = torch.zeros_like(metric)
    count = 0

    def jac(z):
        return (span.log() - F.softplus(z) - F.softplus(-z)).sum(-1)

    def proposal_gradient(z):
        with torch.enable_grad():
            v = z.detach().requires_grad_()
            xx = lo + span * torch.sigmoid(v)
            lp = physical(xx) + strength * guide(xx) + jac(v)
            g = torch.autograd.grad(lp.sum(), v)[0]
        valid = torch.isfinite(lp) & torch.isfinite(g).all(-1)
        return (torch.nan_to_num(g.detach()), valid)

    def true_density(z):
        with torch.no_grad():
            lp = physical(lo + span * torch.sigmoid(z)) + jac(z)
        return torch.where(torch.isfinite(lp), lp, -torch.inf)

    def velocity(momentum):
        return (metric @ momentum[..., None]).squeeze(-1)

    def kinetic(momentum):
        return 0.5 * (momentum * velocity(momentum)).sum(-1)
    samples = torch.empty((steps, chains, dim), dtype=x.dtype)
    logprob = torch.empty((steps, chains), dtype=x.dtype)
    accepted = torch.zeros((steps, chains), dtype=torch.bool)
    errors = torch.empty((steps, chains), dtype=x.dtype)
    gradient, valid = proposal_gradient(q)
    true_lp = true_density(q)
    if not bool(valid.all() & torch.isfinite(true_lp).all()):
        raise ValueError('Initial physical/guide density must be finite')
    with torch.no_grad():
        for iteration, nleap in enumerate(lengths):
            z = torch.randn((chains, dim), generator=rng, dtype=x.dtype)
            momentum = torch.linalg.solve_triangular(chol.transpose(-1, -2), z[..., None], upper=True).squeeze(-1)
            old_energy = -true_lp + kinetic(momentum)
            proposal = q.clone()
            p = momentum + 0.5 * eps[:, None] * gradient
            alive = torch.ones(chains, dtype=torch.bool)
            for k in range(int(nleap)):
                proposal += eps[:, None] * velocity(p)
                new_gradient, ok = proposal_gradient(proposal)
                alive &= ok & torch.isfinite(proposal).all(-1)
                p += (1.0 if k + 1 < nleap else 0.5) * eps[:, None] * new_gradient
            p = -p
            new_true_lp = true_density(proposal)
            delta = -new_true_lp + kinetic(p) - old_energy
            alpha = torch.where(alive & torch.isfinite(delta), torch.exp(torch.minimum(-delta, torch.zeros_like(delta))), 0.0)
            take = torch.rand(chains, generator=rng, dtype=x.dtype) < alpha
            q = torch.where(take[:, None], proposal, q)
            gradient = torch.where(take[:, None], new_gradient, gradient)
            true_lp = torch.where(take, new_true_lp, true_lp)
            samples[iteration] = lo + span * torch.sigmoid(q)
            logprob[iteration] = true_lp - jac(q)
            accepted[iteration] = take
            errors[iteration] = delta
            if iteration < burn_in:
                dual += 1
                rate = 1 / (dual + 10.0)
                hbar = (1 - rate) * hbar + rate * (target_accept - alpha)
                logeps = (mu - math.sqrt(dual) / 0.05 * hbar).clamp(math.log(1e-06), math.log(10.0))
                weight = dual ** (-0.75)
                logbar = weight * logeps + (1 - weight) * logbar
                eps = logeps.exp()
                if iteration >= 50 and iteration < burn_in - 50:
                    count += 1
                    diff = q - mean
                    mean += diff / count
                    m2 += diff[..., None] * (q - mean)[:, None, :]
                if iteration + 1 in updates and count > dim + 1:
                    metric = count / (count + 5.0) * m2 / (count - 1) + 0.001 * 5 / (count + 5.0) * eye
                    metric = 0.5 * (metric + metric.transpose(-1, -2))
                    chol = torch.linalg.cholesky(metric)
                    mean = q.clone()
                    m2.zero_()
                    count = 0
                    eps = logbar.exp()
                    mu = (10 * eps).log()
                    hbar.zero_()
                    dual = 0
                if iteration + 1 == burn_in:
                    eps = logbar.exp()
    divergent = ~torch.isfinite(errors) | (errors.abs() > 1000.0)
    return dict(chain=samples, posterior=samples[burn_in:], log_prob=logprob, acceptance=accepted[burn_in:].to(x.dtype).mean(0), production_divergences=divergent[burn_in:].sum(0), energy_error=errors, step_size=eps, inverse_mass=metric, gradient_evaluations=1 + int(lengths.sum()), physical_endpoint_evaluations=steps + 1, steps=steps, burn_in=burn_in, strength=strength, seed=seed, algorithm='Frozen neural/physical proposal potential; exact physical Hamiltonian Metropolis correction; one box Jacobian')

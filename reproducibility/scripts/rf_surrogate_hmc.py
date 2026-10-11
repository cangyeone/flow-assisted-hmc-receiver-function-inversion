# Our CPU-only archived research implementation; experimental, not default inference.
"""Surrogate-only proposal gradients; exact physical Hamiltonian correction.

Neither data duplication nor network scores enter the sampled target. No
training or potential changes are permitted during production.
"""
import math
import numpy as np
import torch
from rf_physics_surrogate import box_jacobian

def surrogate_hmc_sample(physical, proposal_z, initial, lower, upper, *, steps=2200, burn_in=800, leapfrog_steps=8, target_accept=0.9, seed=0, device='cpu', inverse_mass=None, step_size=0.2, progress=None, state_refresh=None):
    dev = torch.device(device)
    x = torch.as_tensor(initial, dtype=torch.float64, device=dev).clone()
    lo = torch.as_tensor(lower, dtype=x.dtype, device=dev)
    hi = torch.as_tensor(upper, dtype=x.dtype, device=dev)
    span = hi - lo
    if x.ndim != 2 or lo.shape != (x.shape[1],) or hi.shape != lo.shape:
        raise ValueError('Invalid state/bounds shape')
    if not bool(torch.isfinite(x).all() & (hi > lo).all() & (x > lo).all() & (x < hi).all()):
        raise ValueError('Require interior initial states')
    if steps <= burn_in or burn_in < 1 or leapfrog_steps < 1:
        raise ValueError('Invalid iterations')
    chains, dim = x.shape
    q = torch.logit((x - lo) / span)
    eye = torch.eye(dim, dtype=x.dtype, device=dev).expand(chains, dim, dim)
    metric = eye.clone() if inverse_mass is None else torch.broadcast_to(torch.as_tensor(inverse_mass, dtype=x.dtype, device=dev), (chains, dim, dim)).clone()
    chol = torch.linalg.cholesky(metric)
    rng = torch.Generator(device=dev).manual_seed(seed)
    lengths = np.random.default_rng(seed).integers(max(1, leapfrog_steps // 2), 3 * leapfrog_steps // 2 + 1, steps)
    eps = torch.full((chains,), step_size, dtype=x.dtype, device=dev)
    logbar = eps.log()
    mu = (10 * eps).log()
    hbar = torch.zeros_like(eps)
    dual = 0

    def gradient(z):
        with torch.enable_grad():
            v = z.detach().requires_grad_()
            lp = proposal_z(v)
            g = torch.autograd.grad(lp.sum(), v)[0]
        valid = torch.isfinite(lp) & torch.isfinite(g).all(-1)
        return (torch.nan_to_num(g.detach()), valid)

    def true_density(z):
        with torch.no_grad():
            lp = physical(lo + span * torch.sigmoid(z)) + box_jacobian(z, span)
        return torch.where(torch.isfinite(lp), lp, -torch.inf)

    def velocity(p):
        return (metric @ p[..., None]).squeeze(-1)

    def kinetic(p):
        return 0.5 * (p * velocity(p)).sum(-1)
    samples = torch.empty((steps, chains, dim), dtype=x.dtype, device=dev)
    densities = torch.empty((steps, chains), dtype=x.dtype, device=dev)
    accepted = torch.zeros((steps, chains), dtype=torch.bool, device=dev)
    errors = torch.empty_like(densities)
    grad, valid = gradient(q)
    lp = true_density(q)
    if not bool(valid.all() & torch.isfinite(lp).all()):
        raise ValueError('Invalid initial density')
    with torch.no_grad():
        for iteration, nleap in enumerate(lengths):
            z = torch.randn((chains, dim), dtype=x.dtype, device=dev, generator=rng)
            momentum = torch.linalg.solve_triangular(chol.transpose(-1, -2), z[..., None], upper=True).squeeze(-1)
            old_energy = -lp + kinetic(momentum)
            candidate = q.clone()
            p = momentum + 0.5 * eps[:, None] * grad
            alive = torch.ones(chains, dtype=torch.bool, device=dev)
            for k in range(int(nleap)):
                candidate += eps[:, None] * velocity(p)
                new_grad, ok = gradient(candidate)
                alive &= ok & torch.isfinite(candidate).all(-1)
                p += (1.0 if k + 1 < nleap else 0.5) * eps[:, None] * new_grad
            p = -p
            new_lp = true_density(candidate)
            delta = -new_lp + kinetic(p) - old_energy
            alpha = torch.where(alive & torch.isfinite(delta), torch.exp(torch.minimum(-delta, torch.zeros_like(delta))), 0.0)
            take = torch.rand(chains, dtype=x.dtype, device=dev, generator=rng) < alpha
            q = torch.where(take[:, None], candidate, q)
            grad = torch.where(take[:, None], new_grad, grad)
            lp = torch.where(take, new_lp, lp)
            if state_refresh is not None:
                refreshed = state_refresh(lo + span * torch.sigmoid(q))
                q = torch.logit((refreshed - lo) / span)
                grad, valid = gradient(q)
                lp = true_density(q)
            samples[iteration] = lo + span * torch.sigmoid(q)
            densities[iteration] = lp - box_jacobian(q, span)
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
                if iteration + 1 == burn_in:
                    eps = logbar.exp()
            if progress and ((iteration + 1) % 500 == 0 or iteration + 1 == steps):
                progress(iteration + 1, steps)
    divergences = ~torch.isfinite(errors) | (errors.abs() > 1000)
    return dict(chain=samples, posterior=samples[burn_in:], log_prob=densities, acceptance=accepted[burn_in:].to(x.dtype).mean(0), production_divergences=divergences[burn_in:].sum(0), energy_error=errors, step_size=eps, inverse_mass=metric, gradient_evaluations=1 + int(lengths.sum()), physical_gradient_evaluations=0, physical_endpoint_evaluations=steps + 1 + (steps if state_refresh is not None else 0) + 0, seed=seed, steps=steps, burn_in=burn_in, algorithm='Frozen surrogate proposal force, exact physical Hamiltonian acceptance, one box Jacobian')

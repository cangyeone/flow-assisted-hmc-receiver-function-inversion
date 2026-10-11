"""Transformer RF distribution and exact conditional coupling flow.

Exact refers to the invertible learned density, not exact physical posterior.
No learned forward prediction is used to create inverse training targets.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .structure import active_mask, decode

NT, CHANNELS, PATCH = 151, 9, 5
NPATCH = math.ceil(NT / PATCH)
PMIN = torch.tensor([.039, .051, .067])


def position(x, width):
    f = torch.exp(torch.arange(width // 2, device=x.device, dtype=x.dtype) * (-math.log(10000.) / (width // 2)))
    a = x[..., None] * f
    return torch.cat((a.sin(), a.cos()), -1)


def conditions(p, p_sd, sigma):
    return torch.cat(((p - .06) / .02, p_sd / .003,
                      (sigma.clamp_min(1e-6).log() - math.log(.012)) / 1.), -1)


def encoder(width, depth, heads=4):
    # No dropout inside a flow: a deterministic conditioner is required for
    # invertibility, in evaluation AND in training.
    layer = nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.,
                                       activation='gelu', batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class ForwardTransformer(nn.Module):
    def __init__(self, width=96, depth=3, std_min=.001, std_max=.03):
        super().__init__()
        self.width, self.std_min, self.std_max = width, std_min, std_max
        self.layer_in = nn.Linear(8, width)
        self.layer_encoder = encoder(width, depth)
        self.context_in = nn.Linear(10, width)
        layer = nn.TransformerDecoderLayer(width, 4, width * 2, dropout=0.,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.mean = nn.Linear(width, CHANNELS * PATCH)
        self.variance = nn.Linear(width, CHANNELS * PATCH)
        nn.init.zeros_(self.variance.weight)
        nn.init.constant_(self.variance.bias, -1.5)

    def forward(self, layers, lengths, p, p_sd, sigma, tau):
        # lengths INCLUDE the halfspace; masked padding contributes nothing.
        k = layers.shape[1]
        valid = torch.arange(k, device=layers.device)[None] < lengths[:, None]
        top = torch.where(valid, layers[..., 0], 0.)
        bottom = torch.cat((top[:, 1:], top[:, -1:]), -1)
        h = (bottom - top).clamp_min(0.)
        half = torch.arange(k, device=layers.device)[None] == lengths[:, None] - 1
        h = torch.where(half, 0., h)
        q = torch.where(valid[..., None], layers, 0.)
        # Reciprocal Q handles elastic halfspace (infinite Q) without NaNs.
        iq = torch.where((q[..., 4:] > 0) & torch.isfinite(q[..., 4:]),
                         q[..., 4:].clamp_min(1.).reciprocal(), 0.)
        features = torch.stack((top / 80., h / 20., q[..., 1] / 8., q[..., 2] / 4.,
                                q[..., 3] / 3., iq[..., 0] * 50., iq[..., 1] * 50., half.to(q.dtype)), -1)
        memory = self.layer_encoder(self.layer_in(features) + position(top / 2., self.width),
                                    src_key_padding_mask=~valid)
        c = torch.cat((conditions(p, p_sd, sigma), tau[:, None] / .3), -1)
        pos = torch.arange(NPATCH, device=layers.device, dtype=layers.dtype) * PATCH * .2
        query = position(pos, self.width)[None] + self.context_in(c)[:, None]
        out = self.decoder(query, memory, memory_key_padding_mask=~valid)
        mean = self.mean(out).reshape(len(layers), NPATCH, CHANNELS, PATCH).permute(0, 2, 1, 3).reshape(-1, CHANNELS, NPATCH * PATCH)[..., :NT]
        std = self.std_min + (self.std_max - self.std_min) * torch.sigmoid(self.variance(out).reshape(len(layers), NPATCH, CHANNELS, PATCH).permute(0, 2, 1, 3).reshape(-1, CHANNELS, NPATCH * PATCH)[..., :NT])
        anchor = torch.arange(NT, device=layers.device) == 10
        mean = torch.where(anchor, 1., mean)
        std = torch.where(anchor, 0., std)
        return mean, std


class RFTransformer(nn.Module):
    def __init__(self, width=96, depth=3):
        super().__init__()
        self.width = width
        self.patch_in = nn.Linear(CHANNELS * PATCH, width)
        self.context_in = nn.Linear(9, width)
        self.transformer = encoder(width, depth)
        self.norm = nn.LayerNorm(width)

    def forward(self, rf, p, p_sd, sigma):
        x = F.pad(rf, (0, NPATCH * PATCH - NT)).reshape(len(rf), CHANNELS, NPATCH, PATCH).permute(0, 2, 1, 3).flatten(2)
        pos = torch.arange(NPATCH, device=rf.device, dtype=rf.dtype) * PATCH * .2
        x = self.patch_in(x) + position(pos, self.width)[None] + self.context_in(conditions(p, p_sd, sigma))[:, None]
        return self.norm(self.transformer(x).mean(1))


class Coupling(nn.Module):
    def __init__(self, max_layers, width, parity):
        super().__init__()
        self.width = width
        slot = torch.arange(max_layers + 1)[:, None]
        feature = torch.arange(5)[None]
        mask = ((slot + feature + parity) % 2 == 0).float()
        self.register_buffer('keep', mask)
        self.input = nn.Linear(11, width)  # values, keep flags, active-token flag
        self.context = nn.Linear(width, width)
        self.transformer = encoder(width, 1)
        self.output = nn.Linear(width, 10)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def parameters_at(self, x, active, context):
        keep = self.keep[None] * active
        tokens = active.any(-1)
        inputs = torch.cat((x * keep, keep.expand_as(x), tokens[..., None].to(x.dtype)), -1)
        pos = position(torch.arange(x.shape[1], device=x.device, dtype=x.dtype), self.width)[None]
        encoded = self.transformer(self.input(inputs) + pos + self.context(context)[:, None],
                                   src_key_padding_mask=~tokens)
        shift, scale = self.output(encoded).chunk(2, -1)
        change = active * (1 - self.keep[None])
        # Bounded log-scales prevent exponentiation overflow. Smooth affine
        # couplings can still represent multimodal conditional distributions.
        return 2. * torch.tanh(shift / 2.) * change, 1.2 * torch.tanh(scale / 1.2) * change

    def forward(self, x, active, context, inverse=False):
        shift, scale = self.parameters_at(x, active, context)
        y = (x - shift) * torch.exp(-scale) if inverse else x * torch.exp(scale) + shift
        return y * active, (-scale if inverse else scale).sum((1, 2))


class PosteriorTransformerFlow(nn.Module):
    def __init__(self, max_layers=16, min_layers=2, width=96, depth=3, blocks=8):
        super().__init__()
        self.max_layers, self.min_layers = max_layers, min_layers
        self.rf_encoder = RFTransformer(width, depth)
        self.count_head = nn.Linear(width, max_layers - min_layers + 1)
        self.count_embedding = nn.Embedding(max_layers + 1, width)
        self.flow = nn.ModuleList(Coupling(max_layers, width, i % 2) for i in range(blocks))
        self.register_buffer('u_center', torch.zeros(max_layers + 1, 5))
        self.register_buffer('u_scale', torch.ones(max_layers + 1, 5))

    def log_prob(self, u, count, rf, p, p_sd, sigma):
        if bool(((count < self.min_layers) | (count > self.max_layers)).any()):
            raise ValueError('Count outside learned support')
        context = self.rf_encoder(rf, p, p_sd, sigma)
        logk = F.log_softmax(self.count_head(context), -1).gather(1, (count - self.min_layers)[:, None]).squeeze(-1)
        c = context + self.count_embedding(count)
        active = active_mask(count, self.max_layers, u.dtype)
        z = (u - self.u_center) / self.u_scale * active
        jac = -(self.u_scale.log()[None] * active).sum((1, 2))
        for block in self.flow:
            z, j = block(z, active, c)
            jac = jac + j
        log_base = (-.5 * (z.square() + math.log(2 * math.pi)) * active).sum((1, 2))
        return logk + log_base + jac, logk

    @torch.no_grad()
    def sample(self, rf, p, p_sd, sigma, draws=256, count=None):
        context = self.rf_encoder(rf, p, p_sd, sigma)
        probs = torch.softmax(self.count_head(context), -1)
        k = torch.multinomial(probs, draws, replacement=True).flatten() + self.min_layers if count is None else torch.full((len(rf) * draws,), int(count), device=rf.device, dtype=torch.long)
        if bool(((k < self.min_layers) | (k > self.max_layers)).any()):
            raise ValueError('Count outside learned support')
        context = context.repeat_interleave(draws, 0) + self.count_embedding(k)
        active = active_mask(k, self.max_layers, rf.dtype)
        z = torch.randn_like(active) * active
        for block in reversed(self.flow):
            z, _ = block(z, active, context, inverse=True)
        z = (z * self.u_scale + self.u_center) * active
        layers, tau, lj = decode(z, k)
        return dict(u=z, count=k, layers=layers, tau=tau, coordinate_log_jacobian=lj,
                    count_probabilities=probs)


def forward_loss(mean, std, clean, observed, true_std, hetero_weight):
    valid = torch.arange(NT, device=mean.device) != 10
    residual = mean[..., valid] - clean[..., valid]
    # Clean-waveform MSE is always retained. beta-NLL (beta=.5, detached
    # weights) stabilizes heteroscedastic optimization. Std supervision uses
    # the declared simulator noise, not the emulator's own residual target.
    mse = residual.square().mean()
    v = std[..., valid].square().clamp_min(1e-6)
    nll = .5 * ((observed[..., valid] - mean[..., valid]).square() / v + v.log())
    beta_nll = (v.detach().sqrt() * nll).mean()
    var_error = (std[..., valid].clamp_min(.001).log() - true_std[..., valid].clamp_min(.001).log()).square().mean()
    derivative = (mean[..., 1:] - mean[..., :-1] - clean[..., 1:] + clean[..., :-1]).square().mean()
    total = 20 * mse + 5 * derivative + hetero_weight * (.1 * beta_nll + .01 * var_error)
    return total, dict(mse=mse.detach(), derivative=derivative.detach(), beta_nll=beta_nll.detach(), variance_log_mse=var_error.detach())

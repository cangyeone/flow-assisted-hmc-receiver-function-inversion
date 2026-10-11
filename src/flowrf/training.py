"""CPU training with source-disjoint simulation and bounded heteroscedasticity."""
import argparse,json,math,os,time,platform
from pathlib import Path
import numpy as np
import torch
from .neural.data import generate,load_arrays,Noise,save_json,sha
from .neural.models import ForwardTransformer,PosteriorTransformerFlow,forward_loss
from .neural.structure import active_mask
def model_for(arm, c):
    if arm == 'forward':
        return ForwardTransformer(c['width'], c['depth'], c['std_min'], c['std_max'])
    return PosteriorTransformerFlow(c['max_layers'], c['min_layers'], c['width'], c['depth'], c['blocks'])

def device_data(data, device):
    return {k: torch.as_tensor(v, device=device, dtype=torch.long if k == 'count' else torch.float32) for k, v in data.items() if k not in ('source_id', 'context_cluster')}

def initialize_scale(model, data, c):
    a = active_mask(data['count'], c['max_layers'])
    number = a.sum(0).clamp_min(1.0)
    center = (data['u'] * a).sum(0) / number
    scale = (((data['u'] - center).square() * a).sum(0) / number).sqrt().clamp_min(0.25)
    model.u_center.copy_(center)
    model.u_scale.copy_(scale)

def batch_loss(model, arm, data, idx, noise, generator, hetero_weight):
    observed, true_std = noise.draw(data['clean'][idx], data['sigma'][idx], generator)
    if arm == 'forward':
        mean, std = model(data['layers'][idx], data['count'][idx] + 1, data['p'][idx], data['p_sd'][idx], data['sigma'][idx], data['tau'][idx])
        return forward_loss(mean, std, data['clean'][idx], observed, true_std, hetero_weight)
    logp, logk = model.log_prob(data['u'][idx], data['count'][idx], observed, data['p'][idx], data['p_sd'][idx], data['sigma'][idx])
    return (-logp.mean(), dict(nll=-logp.detach().mean(), count_nll=-logk.detach().mean()))

@torch.no_grad()
def validate(model, arm, data, noise, c, epoch, device):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(c['noise_seed'] + 100)
    total, mse, count_nll, n = (0.0, 0.0, 0.0, len(data['clean']))
    for first in range(0, n, c['batch']):
        idx = slice(first, min(first + c['batch'], n))
        value, metrics = batch_loss(model, arm, data, idx, noise, generator, 1.0)
        num = len(data['clean'][idx])
        total += float(value) * num
        if arm == 'forward':
            mse += float(metrics['mse']) * num
        else:
            count_nll += float(metrics['count_nll']) * num
    return dict(objective=total / n, rf_rmse=math.sqrt(mse / n)) if arm == 'forward' else dict(joint_nll=total / n, count_nll=count_nll / n)

def checkpoint(path, value):
    temp = path.with_suffix('.tmp.pt')
    torch.save(value, temp)
    os.replace(temp, path)

def train(folder, stage, arm, count, epochs, c, device, initial=None):
    split = 'pilot' if stage == 'pilot' else 'train'
    output = Path(folder) / stage / arm
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').exists():
        return
    seed = c[f'{arm}_seed']
    torch.manual_seed(seed)
    model = model_for(arm, c).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    begin = time.perf_counter()
    arrays = load_arrays(folder, split, count)
    if len(arrays['clean']) != count:
        raise RuntimeError('Training count is not complete')
    data = device_data(arrays, device)
    del arrays
    valid = device_data(load_arrays(folder, 'validation'), device)
    noise = Noise(device)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 10000)
    order_generator = torch.Generator(device=device).manual_seed(seed + 20000)
    start_epoch, best, history = (0, math.inf, [])
    latest = output / 'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved['config_sha256'] != c['config_sha256']:
            raise ValueError('Checkpoint/config mismatch')
        model.load_state_dict(saved['state_dict'])
        opt.load_state_dict(saved['optimizer'])
        start_epoch, best, history = (saved['epoch'], saved['best'], saved['history'])
        noise_generator.set_state(saved['noise_rng'].cpu())
        order_generator.set_state(saved['order_rng'].cpu())
        torch.set_rng_state(saved['torch_rng'].cpu())
    elif initial is not None:
        saved = torch.load(initial, map_location=device, weights_only=True)
        model.load_state_dict(saved['state_dict'])
    elif arm == 'inverse':
        initialize_scale(model, data, c)
    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = torch.randperm(count, device=device, generator=order_generator)
        total, maximum_gradient = (0.0, 0.0)
        hetero = 0.0 if epoch <= c['forward_warmup_epochs'] else min(1.0, (epoch - c['forward_warmup_epochs']) / c['hetero_ramp_epochs'])
        if stage == 'pilot':
            hetero = 1.0 if epoch == epochs else 0.0
        lr = c['learning_rate'] * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / epochs)))
        for group in opt.param_groups:
            group['lr'] = lr
        for first in range(0, count, c['batch']):
            idx = order[first:first + c['batch']]
            value, _ = batch_loss(model, arm, data, idx, noise, noise_generator, hetero)
            if not bool(torch.isfinite(value)):
                raise FloatingPointError(f'{stage}/{arm} nonfinite loss at {epoch}:{first}')
            opt.zero_grad(set_to_none=True)
            value.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            maximum_gradient = max(maximum_gradient, float(grad))
            opt.step()
            total += float(value.detach()) * len(idx)
            if first == 0 or first // c['batch'] % 100 == 0:
                save_json(Path(folder) / f'{stage}-{arm}-status.json', dict(stage=stage, arm=arm, device=device, epoch=epoch, epochs=epochs, completed_in_epoch=first + len(idx), training_sources=count, loss=float(value.detach()), hetero_weight=hetero, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        scores = validate(model, arm, valid, noise, c, epoch, device)
        criterion = scores['rf_rmse'] if arm == 'forward' else scores['joint_nll']
        entry = dict(stage=stage, arm=arm, epoch=epoch, train_loss=total / count, validation=scores, seconds=time.perf_counter() - epoch_start, max_unclipped_gradient=maximum_gradient, lr=lr, hetero_weight=hetero)
        history.append(entry)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        metadata = dict(arm=arm, stage=stage, selected_epoch=epoch, validation=scores, device=device, trained_sources=count, parameter_count=sum((p.numel() for p in model.parameters())), config_sha256=c['config_sha256'], configuration=c, semantics='approximate physical simulator posterior' if arm == 'inverse' else 'physical mean RF with bounded specified-noise std')
        eligible = not (stage == 'full' and arm == 'forward' and (epochs > 10)) or hetero >= 1.0
        if eligible and criterion < best:
            best = criterion
            checkpoint(output / 'best.pt', dict(state_dict=state, metadata=metadata))
        checkpoint(latest, dict(state_dict=state, optimizer=opt.state_dict(), epoch=epoch, best=best, history=history, config_sha256=c['config_sha256'], noise_rng=noise_generator.get_state(), order_rng=order_generator.get_state(), torch_rng=torch.get_rng_state()))
        save_json(output / 'training.json', dict(history=history, best=best if math.isfinite(best) else None, train_seconds_this_process=time.perf_counter() - begin))
        print(json.dumps(entry), flush=True)
    save_json(output / 'complete.json', dict(epochs=epochs, trained_sources=count, best=best, checkpoint_sha256=sha(output / 'best.pt'), config_sha256=c['config_sha256']))

def main():
    ap=argparse.ArgumentParser(description='CPU simulation / training; long runs are explicit')
    ap.add_argument('--output',required=True);ap.add_argument('--config',required=True)
    ap.add_argument('--stage',choices=['generate','pilot','bootstrap','full'],default='generate')
    ap.add_argument('--arm',choices=['forward','inverse','both'],default='both');ap.add_argument('--threads',type=int,default=1)
    a=ap.parse_args();torch.set_num_threads(a.threads)
    folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True)
    c=json.loads(Path(a.config).read_text());f=folder/'configuration.json'
    if f.exists() and json.loads(f.read_text())!=c:raise ValueError('Frozen configuration changed; use a new output directory')
    save_json(f,c)
    import grtm
    import grtm.rf_cpu as physical
    source_root=Path(__file__).parent
    protocol=dict(configuration=c,sources={name:sha(source_root/name) for name in ['training.py','neural/data.py','neural/models.py','neural/structure.py']},physical_module_sha256=sha(physical.__file__),runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),grtm=grtm.__version__,device='cpu',threads=torch.get_num_threads()))
    frozen=folder/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text())!=protocol:raise ValueError('Frozen sources or runtime changed; use a new output directory')
    save_json(frozen,protocol);c['config_sha256']=sha(frozen)
    if a.stage=='generate':
        for split in ['pilot','train','validation','test']:generate(folder,split,c[split],c,'cpu')
    else:
        count={'pilot':c['pilot'],'bootstrap':c['bootstrap'],'full':c['train']}[a.stage]
        for arm in ['forward','inverse'] if a.arm=='both' else [a.arm]:
            initial=None
            if a.stage!='pilot':
                previous='pilot' if a.stage=='bootstrap' else 'bootstrap'
                initial=folder/previous/arm/'best.pt'
                if not initial.exists():raise FileNotFoundError('Run '+previous+' training first')
            epochs=c['pilot_epochs'] if a.stage=='pilot' else c['bootstrap_epochs'] if a.stage=='bootstrap' else c[arm+'_epochs']
            train(folder,a.stage,arm,count,epochs,c,'cpu',initial)

if __name__=='__main__':main()

import argparse, json, time
from pathlib import Path
import torch
from ma_psdun.real_data import load_real_dataset
from ma_psdun.core import MeasurementOperator
from ma_psdun.model import MAPSDUN, loss_fn
from ma_psdun.eval import image_metrics
from ma_psdun.conditions import OD_TRANSMITTANCE

OD_TO_S = OD_TRANSMITTANCE

def make_batch(samples, indices, device):
    np = __import__('numpy')
    # `load_real_dataset` already forms a per-capture normalized dark-
    # subtracted measurement.  Feed that calibrated signal to the TCM as the
    # bright channel and use an explicit zero dark channel; passing raw ADC
    # values here would undo the calibration through a large scale mismatch.
    raw = torch.from_numpy(np.stack([samples[i]['y'] for i in indices])).to(device)
    dark = torch.zeros_like(raw)
    target = torch.from_numpy(__import__('numpy').stack([samples[i]['label'] for i in indices])).to(device)[:, None]
    intensity = torch.tensor([OD_TO_S[samples[i]['od']] for i in indices], device=device)
    cond = torch.stack([torch.ones_like(intensity), intensity, torch.full_like(intensity, 550.)], -1)
    return raw[:, None], dark[:, None], target, cond

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--patterns',required=True); p.add_argument('--exp-dir',required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--epochs',type=int,default=300); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--stages',type=int,default=5); p.add_argument('--seed',type=int,default=1001); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    import numpy as np
    torch.manual_seed(a.seed); device=torch.device(a.device if torch.cuda.is_available() else 'cpu'); out=Path(a.exp_dir); out.mkdir(parents=True,exist_ok=True)
    patterns,samples=load_real_dataset(a.data_root,a.patterns); patterns=torch.from_numpy(patterns)
    n=patterns.shape[1]; op=MeasurementOperator(n,n,device=device,patterns=patterns); model=MAPSDUN(op,stages=a.stages,backprojection_gain_init=4.0).to(device); opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    groups={'train':[i for i,s in enumerate(samples) if s['object'] in ('bar1-20260830','bar2-20260830')], 'val':[i for i,s in enumerate(samples) if s['object']=='bar3-20260830'], 'test':[i for i,s in enumerate(samples) if s['object']=='bar4-20260830']}
    (out/'split.json').write_text(json.dumps({k:[samples[i]['object']+'/'+samples[i]['od'] for i in v] for k,v in groups.items()},indent=2))
    (out/'config.json').write_text(json.dumps(vars(a),indent=2)); best_mse=float('inf'); best_ssim=-float('inf'); log=[]
    train_raw,train_dark,train_target,train_cond=make_batch(samples,groups['train'],device); val_raw,val_dark,val_target,val_cond=make_batch(samples,groups['val'],device)
    epochs=10 if a.smoke else a.epochs
    for epoch in range(epochs):
        model.train(); pred,y=model(train_raw,train_dark,train_cond,(64,64)); loss=loss_fn(pred,train_target,y,op); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp,vy=model(val_raw,val_dark,val_cond,(64,64)); vm=image_metrics(vp,val_target); vm['epoch']=epoch; vm['train_loss']=float(loss.detach())
            vm['corr']=float(torch.corrcoef(torch.stack([vp.flatten(),val_target.flatten()]))[0,1])
            vm['pred_std']=float(vp.std()); vm['target_std']=float(val_target.std())
        log.append(vm); print(json.dumps(vm),flush=True)
        if vm['mse']<best_mse:
            best_mse=vm['mse']; torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'validation':vm},out/'checkpoint_best_mse.pt')
        if vm['ssim']>best_ssim:
            best_ssim=vm['ssim']; torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'validation':vm},out/'checkpoint_best.pt')
        torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'validation':vm},out/'checkpoint_latest.pt')
    model.load_state_dict(torch.load(out/'checkpoint_best.pt',map_location=device)['model']); model.eval()
    with torch.no_grad():
        test_batch=make_batch(samples,groups['test'],device)
        tp,ty=model(test_batch[0],test_batch[1],test_batch[3],(64,64)); tm=image_metrics(tp,test_batch[2])
        tm['corr']=float(torch.corrcoef(torch.stack([tp.flatten(),test_batch[2].flatten()]))[0,1])
        tm['pred_std']=float(tp.std()); tm['target_std']=float(test_batch[2].std()); tm['checkpoint_metric']='ssim'
    (out/'validation.jsonl').write_text('\n'.join(json.dumps(x) for x in log)+'\n'); (out/'test.json').write_text(json.dumps(tm,indent=2)); torch.save({'pred':tp.cpu(),'target':make_batch(samples,groups['test'],device)[2].cpu()},out/'test_samples.pt'); (out/'DONE').write_text('completed\n')
    (out/'experiment.md').write_text('# Real MA-PSDUN training\n\n- labels: SI_AP.png reference reconstruction, not physical ground truth\n- split: bar1/bar2 train, bar3 validation, bar4 test\n- input: normalized dark-subtracted y from dark/bright pairs in traindata.txt\n- pattern TIFF: binary 0/1 DMD random-speckle frames\n- operator: centered 0/1 pattern operator, scale 1/sqrt(M)\n- checkpoint selection: best validation SSIM (MSE checkpoint also saved)\n- best validation MSE: %.8f\n- best validation SSIM: %.8f\n- test MSE: %.8f\n- test PSNR: %.6f\n- test SSIM: %.6f\n- test correlation: %.6f\n' % (best_mse,best_ssim,tm['mse'],tm['psnr'],tm['ssim'],tm['corr']))

if __name__=='__main__': main()

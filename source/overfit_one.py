import argparse, json
from pathlib import Path
import torch
from ma_psdun.core import MeasurementOperator, structured_images, simulate_measurements
from ma_psdun.model import MAPSDUN, loss_fn
from ma_psdun.eval import image_metrics

p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--steps',type=int,default=500); a=p.parse_args()
device=torch.device(a.device if torch.cuda.is_available() else 'cpu'); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
op=MeasurementOperator(128*128,3276,seed=123,device=device); model=MAPSDUN(op,stages=5).to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
target=structured_images(1,128,seed=999).to(device); intensity=torch.tensor([0.1],device=device); raw,dark=simulate_measurements(target,op,intensity,seed=999); cond=torch.tensor([[.2,.1,550.]],device=device); target4=target.reshape(1,1,128,128)
for step in range(a.steps):
 pred,y=model(raw[:,None],dark[:,None],cond,(128,128)); loss=loss_fn(pred,target4,y,op); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
 if step%50==0 or step==a.steps-1: print(json.dumps({'step':step,'loss':float(loss.detach())}),flush=True)
with torch.no_grad():
 pred,y=model(raw[:,None],dark[:,None],cond,(128,128)); metrics=image_metrics(pred,target4); metrics['corr']=float(torch.corrcoef(torch.stack([pred.flatten(),target4.flatten()]))[0,1]); metrics['pred_std']=float(pred.std()); metrics['target_std']=float(target4.std())
(out/'result.json').write_text(json.dumps(metrics,indent=2)); torch.save({'pred':pred.cpu(),'target':target4.cpu()},out/'samples.pt'); print(json.dumps(metrics))

"""Atomic epoch-boundary recovery, preserving original training schedule."""
import hashlib, json, random
from pathlib import Path
import numpy as np
import torch

def signature(cfg, splits):
    import copy
    c=copy.deepcopy(cfg)
    c.get('experiment',{}).pop('output_root',None)
    c.get('data',{}).pop('root',None)
    subset=c.get('data',{}).pop('train_split_file',None)
    if subset: c['data']['subset_sha256']=hashlib.sha256(Path(subset).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps([c,[(s['dataset'],s['split'],s['sha256']) for s in splits]],sort_keys=True,default=str).encode()).hexdigest()

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])

def save(path, trainer, history, best_value, identity):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    torch.save({'format':'bt_epoch_recovery_v1','identity':identity,'epoch':len(history),
        'model':trainer.model.state_dict(),'optimizer':trainer.optimizer.state_dict(),
        'scheduler':trainer.scheduler.state_dict() if trainer.scheduler else None,
        'scaler':trainer.scaler.state_dict(),'history':history,'best_value':best_value,
        'best_record':trainer.best_record,'best_epoch':trainer.best_epoch,
        'best_model':trainer.best_state_dict,'best_optimizer':trainer.best_optimizer_state,
        'rng':capture_rng()},tmp)
    tmp.replace(path)

def load(path, trainer, identity):
    s=torch.load(path,map_location='cpu',weights_only=False)
    if s.get('format')!='bt_epoch_recovery_v1' or s.get('identity')!=identity:
        raise ValueError('Recovery checkpoint configuration/split mismatch')
    if s['epoch']!=len(s['history']): raise ValueError('Invalid recovery epoch')
    trainer.model.load_state_dict(s['model']); trainer.optimizer.load_state_dict(s['optimizer'])
    if trainer.scheduler: trainer.scheduler.load_state_dict(s['scheduler'])
    trainer.scaler.load_state_dict(s['scaler'])
    trainer.best_record=s['best_record']; trainer.best_epoch=s['best_epoch']; trainer.best_state_dict=s['best_model']; trainer.best_optimizer_state=s['best_optimizer']
    restore_rng(s['rng']); return s['history'],s['best_value']

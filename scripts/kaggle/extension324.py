"""Versioned revision queue: unchanged baseline identities, new proposal identities."""
from copy import deepcopy
from scripts.kaggle.extension300 import SEEDS, NEW_DATASETS

RESIDUAL = {'direct','additive_a025','additive_a050','additive_a100','additive_zero_a050','difference_a050','spatial_mixer'}

def proposal_family(spec):
    return spec['model'].startswith('dapr_') or (spec.get('group')=='residual' and spec['model'] in RESIDUAL)

def revise_jobs(previous):
    assert len(previous)==300
    jobs=deepcopy(previous[:288])
    for job in jobs:
        if proposal_family(job):
            job['id']='h2__'+job['id']
            job['fourier_hermitian_constraint']=True
    def add(dataset,model,epochs,seed,group):
        jobs.append(dict(id=f'extension324__{group}__{dataset}__{model}__s{seed}',type='standard',group=group,dataset=dataset,model=model,variant='',config=f'configs/fair/{model}.yaml',seed=seed,epochs=epochs,image_size=352,batch_size=6,num_workers=4))
    # Replace the 12 redundant v2-only comparisons (v2 is now used in all proposal rows).
    for model in ('unet','unetpp','resunetpp','attention_unet'):
        for seed in SEEDS: add('hyper_kvasir_seg',model,30,seed,'matched_hyper_baselines')
    for dataset in NEW_DATASETS:
        epochs=100 if dataset=='cvc_clinicdb' else 30 if dataset=='isic2018' else 50
        for model in ('hsnet','cfanet'):
            for seed in SEEDS: add(dataset,model,epochs,seed,'expanded_baselines')
    assert len(jobs)==324 and len({j['id'] for j in jobs})==324
    return jobs

def apply_model_revision(spec,cfg):
    if spec.get('fourier_hermitian_constraint'):
        if not cfg['model']['name'].startswith('dapr_'):
            raise ValueError('Revision cannot target a non-proposal architecture')
        cfg['model']['fourier_hermitian_constraint']=True
    return cfg

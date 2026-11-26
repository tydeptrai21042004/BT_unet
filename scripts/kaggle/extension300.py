"""Append-only extension; never alter the 165 legacy job identities."""
from copy import deepcopy

NEW_DATASETS = ('cvc_clinicdb', 'cvc_colondb', 'etis', 'isic2018')
MODELS = ('unet','unetpp','attention_unet','resunetpp','pranet','hardnet_mseg','polyp_pvt','linknet','dapr_unet')
SEEDS = (42,1337,2024)

def append_jobs(original):
    assert len(original)==165
    jobs=deepcopy(original)
    def add(dataset,model,seed,epochs,group,variant=''):
        identifier=f'extension300__{group}__{dataset}__{model}__s{seed}'
        config='configs/fair/'+model+'.yaml'
        if variant=='hermitian': config='configs/fair/dapr_hermitian.yaml'
        jobs.append(dict(id=identifier,type='standard',group=group,dataset=dataset,model=model,variant=variant,
                         config=config,seed=seed,epochs=epochs,image_size=352,batch_size=6,num_workers=4))
    for dataset in NEW_DATASETS:
        epochs=100 if dataset=='cvc_clinicdb' else 30 if dataset=='isic2018' else 50
        for model in MODELS:
            for seed in SEEDS: add(dataset,model,seed,epochs,'dataset_expansion')
    for dataset,epochs in [('kvasir_seg',100),('kvasir_instrument',50),('bus_uclm',50),('isbi2012',50),('hyper_kvasir_seg',30)]:
        for seed in SEEDS: add(dataset,'linknet',seed,epochs,'additional_baseline')
    for dataset,epochs in [('kvasir_seg',100),('hyper_kvasir_seg',30),('cvc_clinicdb',100),('isic2018',30)]:
        for seed in SEEDS: add(dataset,'dapr_unet',seed,epochs,'hermitian_comparison','hermitian')
    assert len(jobs)==300 and len({j['id'] for j in jobs})==300
    return jobs

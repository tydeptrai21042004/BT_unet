"""Patient-grouped binary CT/MRI 3D NIfTI/NumPy loader without slice leakage.

Manifest CSV columns: sample_id,patient_id,image,mask. Paths are relative to
--data-root. NIfTI (X,Y,Z) is transposed to tensor (D=Z,H=Y,W=X).
Volume intensities are robustly standardized per volume; no invented labels.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_manifest(path):
    with Path(path).open(newline='',encoding='utf-8') as fh:
        rows=list(csv.DictReader(fh))
    required={'sample_id','patient_id','image','mask'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'{path}: need nonempty CSV columns {sorted(required)}')
    if any(not all(r.get(key,'').strip() for key in required) for r in rows):
        raise ValueError(f'Missing sample/patient/image/mask in {path}')
    ids=[r['sample_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError(f'Duplicate volume IDs in {path}')
    return rows


def audit_patient_splits(train,val,test):
    manifests={'train':read_manifest(train),'val':read_manifest(val),'test':read_manifest(test)}
    groups={name:{r['patient_id'] for r in records} for name,records in manifests.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        shared=groups[a]&groups[b]
        if shared:raise ValueError(f'Patient-level leakage between {a}/{b}: {sorted(shared)[:10]}')
    sample_ids={name:{r['sample_id'] for r in records} for name,records in manifests.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if sample_ids[a]&sample_ids[b]:raise ValueError(f'Duplicate volume IDs across {a}/{b}')
    return {name:{'volumes':len(records),'patients':len(groups[name])} for name,records in manifests.items()}


def load_array(path,kind):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(path)
    name=path.name.lower()
    if name.endswith('.nii') or name.endswith('.nii.gz'):
        try:import nibabel as nib
        except ImportError as exc:raise ImportError('NIfTI volumes require pip install nibabel') from exc
        nii=nib.load(str(path))
        # Return (D,H,W) and physical spacing in the same axis order.
        array=np.asarray(nii.dataobj).transpose(2,1,0)
        spacing=tuple(float(x) for x in nii.header.get_zooms()[:3][::-1])
    elif name.endswith('.npy'):
        # No physical metadata; spacing defaults to unit voxels, NOT millimeters.
        array=np.load(path,allow_pickle=False);spacing=(1.,1.,1.)
    elif name.endswith('.npz'):
        with np.load(path,allow_pickle=False) as data:
            if kind not in data:
                raise KeyError(f'{path}: missing key {kind!r}')
            array=data[kind].copy()
            spacing=tuple(map(float,data['spacing'])) if 'spacing' in data else (1.,1.,1.)
    else:raise ValueError(f'Unsupported 3D volume type: {path}')
    if array.ndim!=3 or any(s<=0 for s in array.shape):
        raise ValueError(f'{path}: expected 3D volume, got {array.shape}')
    if len(spacing)!=3 or not np.isfinite(spacing).all() or min(spacing)<=0:
        raise ValueError(f'{path}: non-positive or missing voxel spacing')
    if not np.isfinite(array).all():raise ValueError(f'{path}: nonfinite values')
    return np.asarray(array), spacing


def crop_or_pad(array,patch,rng=None,center=None):
    # Used ONLY for train patch extraction. Evaluations use native full volumes.
    patch=tuple(map(int,patch))
    if any(k<=0 for k in patch):raise ValueError('Patch sizes must be positive')
    pads=[(0,max(0,p-s)) for s,p in zip(array.shape,patch)]
    if any(p[1] for p in pads):array=np.pad(array,pads,mode='constant')
    starts=[]
    for axis,p in enumerate(patch):
        max_start=array.shape[axis]-p
        starts.append(max(0,min(max_start,round((array.shape[axis]-p)/2))) if rng is None
                      else int(rng.integers(0,max_start+1)))
    slicing=tuple(slice(st,st+p) for st,p in zip(starts,patch))
    return array[slicing],tuple(starts)


class MedicalVolumeDataset(Dataset):
    def __init__(self,data_root,manifest,patch_shape=None,train=False,seed=42,
                 clip_percentiles=(.5,99.5)):
        self.root=Path(data_root).resolve();self.rows=read_manifest(manifest)
        self.patch_shape=tuple(patch_shape) if patch_shape is not None else None
        self.train=bool(train);self.rng=np.random.default_rng(seed)
        self.clip_percentiles=tuple(clip_percentiles)
        if self.train and not self.patch_shape:raise ValueError('Training requires a 3D patch_shape')

    def __len__(self):return len(self.rows)

    def _resolve(self,value):
        path=(self.root/value).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Volume path traverses outside data root')
        return path

    def __getitem__(self,index):
        row=self.rows[index]
        image_path=self._resolve(row['image'])
        mask_path=self._resolve(row['mask'])
        image,spacing=load_array(image_path,'image')
        mask,mask_spacing=load_array(mask_path,'mask')
        nii_image=image_path.name.lower().endswith(('.nii','.nii.gz'))
        nii_mask=mask_path.name.lower().endswith(('.nii','.nii.gz'))
        if nii_image != nii_mask:
            raise ValueError('Volume and mask must use matching NIfTI metadata conventions')
        image_spatial_metadata = bool(nii_image)
        if image_path.name.lower().endswith('.npz'):
            with np.load(image_path,allow_pickle=False) as z:
                image_spatial_metadata = 'spacing' in z.files
        if nii_image:
            import nibabel as nib
            if not np.allclose(nib.load(str(image_path)).affine, nib.load(str(mask_path)).affine,rtol=1e-5,atol=1e-4):
                raise ValueError(f'Volume/mask affine mismatch for {row["sample_id"]}')
        if image.shape!=mask.shape or not np.allclose(spacing,mask_spacing,rtol=1e-4,atol=1e-4):
            raise ValueError(f"Volume/mask geometry mismatch for {row['sample_id']}")
        # Per-volume robust standardization, no pseudo imaging intensities.
        image=image.astype(np.float32,copy=False)
        p0,p1=np.percentile(image,self.clip_percentiles)
        image=np.clip(image,p0,p1)
        image=(image-image.mean())/max(float(image.std()),1e-6)
        binary=(mask>0).astype(np.float32)
        if self.patch_shape:
            # Crop both arrays using EXACT same coordinates.
            pads=[(0,max(0,p-s)) for s,p in zip(image.shape,self.patch_shape)]
            if any(q for _,q in pads):
                image=np.pad(image,pads);binary=np.pad(binary,pads)
            shape=image.shape
            starts=[int(self.rng.integers(0,shape[i]-p+1)) if self.train else
                    (shape[i]-p)//2 for i,p in enumerate(self.patch_shape)]
            cuts=tuple(slice(st,st+p) for st,p in zip(starts,self.patch_shape))
            image=image[cuts];binary=binary[cuts]
        return {'image':torch.from_numpy(np.ascontiguousarray(image[None])).float(),
                'mask':torch.from_numpy(np.ascontiguousarray(binary[None])).float(),
                'id':row['sample_id'],'patient_id':row['patient_id'],
                'spacing':torch.tensor(spacing,dtype=torch.float32),
                'spacing_is_physical':image_spatial_metadata}


__all__=['MedicalVolumeDataset','audit_patient_splits','read_manifest','load_array']

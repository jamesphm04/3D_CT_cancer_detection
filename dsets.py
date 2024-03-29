from collections import namedtuple
import functools
import glob
import csv
import os
import copy
import random
import math

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import SimpleITK as sitk

from utils.util import XyzTuple, xyz2irc
from utils.disk import getCache
from utils.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('raw')

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    ['isNodule_bool', 'diameter_mm', 'series_uid', 'center_xyz']
) #define the type 

@functools.lru_cache(1) # if the same parameters are passed, return result stored in cache
def getCandidateInfoList(requireOnDisk_bool=True):
    #We construct a set with all series_uids that are presen on disk.
    #This will let us use the data even if we havent downloaded all of the subsets yets.
    mhd_list = glob.glob('data/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open('data/annotations.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            
            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )
            
    
    candidateInfo_list = []
    with open('data/candidates.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            
            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz= tuple([float(x) for x in row[1:4]])
            #if delta_mm is less than 1/4 of the diameter, we consider it a match, else diameter is 0.0
            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i]) # the center is not correctly aligned or have not been annotated yet
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break
                
            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool, 
                candidateDiameter_mm, 
                series_uid, 
                candidateCenter_xyz
            ))
    
    candidateInfo_list.sort(reverse=True) #sort by isNodule_bool -> candidateDiameter_mm -> series_uid -> candidateCenter_xyz
    return candidateInfo_list

class Ct: 
    def __init__(self, series_uid):
        print(series_uid)
        mhd_path = glob.glob(f'data/subset*/{series_uid}.mhd')[0]
        
        ct_mhd = sitk.ReadImage(mhd_path) # -> sitk image object
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        
        # CTs are natively expressed on wiki 
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound mukes any weird hotspot and clamps bone down
        ct_a.clip(-1000, 1000, ct_a) # remove all outliers
        
        self.series_uid = series_uid
        self.hu_a = ct_a
        
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin()) # unpack the origin
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
    
    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a
        )      
        
        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])
            
            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
            # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
            #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])
            
            if end_ndx > self.hu_a.shape[axis]:
            # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
            #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])
                
            slice_list.append(slice(start_ndx, end_ndx))
        
        ct_chunk = self.hu_a[tuple(slice_list)]
        
        return ct_chunk, center_irc   
    
@functools.lru_cache(1, typed=True) # repeatly ask for the same Ct instance 
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

def getCtAugmentedCandidate(
    augmentation_dict,
    series_uid, center_xyz, width_irc,
    use_cache=True
):
    if use_cache:
        ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc)
    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
        
    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32) # add batch and channel dimension
    
    transform_t = torch.eye(4) #init a 2d diagonal tensor 4x4
    
    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i, i] *= -1
        
        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i, 3] = offset_float * random_float
        
        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i, i] *= 1.0 + scale_float * random_float
            
    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)
        
        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        transform_t @= rotation_t
        
    affine_t = F.affine_grid(
        transform_t[:3].unsqueeze(0).to(torch.float32),
        ct_t.size(),
        align_corners=False
    )
    
    augmented_chunk = F.grid_sample(
        ct_t,
        affine_t,
        padding_mode='border',
        align_corners=False,
    ).to('cpu')
    
    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']
        
        augmented_chunk += noise_t
    
    return augmented_chunk[0], center_irc
    

class LunaDataset(Dataset):
    # Requirements:
    # - Both train and validation sets should include examples of all variations of expected inputs ---------------------------> equally sample
    # - Neither set should have smaples that aren't representaive of expected inputs 
    #   unless they have a specific purpose like training the model to be robust to outliers ----------------------------------> representative unless on purpose
    # - The training set shouldn't offer unfair hints about the validation set that wouldn't be true for the real-world data
    #   for example include the same sample in both sets, this is knowns as a leak in the training set). ----------------------> training leak
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None, sortby_str='random', ratio_int=0, augmentation_dict=None, candidateInfo_list=None): 
        #val_stride: sampling data every val_stride
        #isValSet_bool: keep only training/validation/everything
        if candidateInfo_list:
            self.candidateInfo_list = copy.copy(candidateInfo_list)  #deep copy
            self.use_cache = False
        else:
            self.candidateInfo_list = copy.copy(getCandidateInfoList())
            self.use_cache = True
        
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict
        
        if series_uid: #if series_uid provided, only return nodules from that ct
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]
            
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride] # partition ot 1/val_stride of the data -> validation data
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride] # delete the validations candidates -> train_data
            assert self.candidateInfo_list
            
        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))
        
        self.negative_list = [
            candidate for candidate in self.candidateInfo_list if not candidate.isNodule_bool 
        ]
        
        self.positive_list = [
            candidate for candidate in self.candidateInfo_list if candidate.isNodule_bool 
        ]
        
        #else -> everything
        log.info(f"{self!r}: {len(self.candidateInfo_list)} {'validation' if isValSet_bool else 'training'} samples")

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.positive_list)

    def __len__(self):
        if self.ratio_int:
            return 500
        else: 
            return len(self.candidateInfo_list)
    
    def __getitem__(self, index):
        # ration positive:negative
        if self.ratio_int:
            pos_index = index // (self.ratio_int + 1)

            if index % (self.ratio_int + 1):
                neg_index = index - 1 - pos_index
                neg_index %= len(self.negative_list)
                candidateInfo_tup = self.negative_list[neg_index]
            else:
                pos_index %= len(self.positive_list)
                candidateInfo_tup = self.positive_list[pos_index]
        else:
            candidateInfo_tup = self.candidateInfo_list[index]
            
        width_irc = (32, 48, 48) #depth, height, width
        
        if self.augmentation_dict:
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.augmentation_dict,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                self.use_cache
            )
        elif self.use_cache:
            candidate_a, center_irc = getCtRawCandidate(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0) # add channel dimension
        else: 
            ct = getCt(candidateInfo_tup.series_uid)
            candidate_a, center_irc = ct.getRawCandidate(
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        
        
        pos_t = torch.tensor([
                not candidateInfo_tup.isNodule_bool,
                candidateInfo_tup.isNodule_bool
            ],
            dtype=torch.long
        ) # CrossEntropyLoss expect one output per class
        
        return (candidate_t, pos_t, candidateInfo_tup.series_uid, torch.tensor(center_irc))

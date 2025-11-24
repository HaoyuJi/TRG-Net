import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose

import math

__all__ = ["ActionSegmentationDataset", "collate_fn"]

dataset_names = ["MCFS-22", "MCFS-130", "PKU-subject", "PKU-view", "LARA", "TCG-15"]
modes = ["training", "validation", "trainval", "test"]

def get_displacements(sample):
    # input: C, T, V, M
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))
    
    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    t = sample.shape[1]
    # Shape: C, t-1, V, M
    disps = sample[:, 1:, :, :] - sample[:, :-1, :, :]
    # Shape: C, T, V, M
    final_sample[:, start:end-1, :, :] = disps

    return final_sample

def get_relative_coordinates(sample,
                             references=(0)):
    # input: C, T, V, M
    # references=(4, 8, 12, 16)
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))
    
    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    C, t, V, M = sample.shape
    rel_coords = []
    #for i in range(len(references)):
    ref_loc = sample[:, :, references, :]
    coords_diff = (sample.transpose((2, 0, 1, 3)) - ref_loc).transpose((1, 2, 0, 3))
    rel_coords.append(coords_diff)
    
    # Shape: C, t, V, M 
    rel_coords = np.vstack(rel_coords)
    # Shape: C, T, V, M
    final_sample[:, start:end, :, :] = rel_coords
    return final_sample

class ActionSegmentationDataset(Dataset): #dataset
    """ Action Segmentation Dataset """

    def __init__(
        self,
        dataset: str,
        transform: Optional[Compose] = None,
        mode: str = "training",
        split: int = 1,
        dataset_dir: str = "./dataset",
        csv_dir: str = "./csv",
        augmentations=False,
    ) -> None:
        super().__init__()
        """
            Args:
                dataset: the name of dataset
                transform: torchvision.transforms.Compose([...])
                mode: training, validation, test
                split: which split of train, val and test do you want to use in csv files.(default:1)
                csv_dir: the path to the directory where the csv files are saved
        """

        assert (
            dataset in dataset_names
        ), "You have to choose dataset."

        if mode == "training":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "train{}.csv".format(split))
            ) # read csv
        elif mode == "validation":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "val{}.csv".format(split))
            )
        elif mode == "trainval":
            df1 = pd.read_csv(
                os.path.join(csv_dir, dataset, "train{}.csv".format(split))
            )
            df2 = pd.read_csv(os.path.join(csv_dir, dataset, "val{}.csv".format(split)))
            self.df = pd.concat([df1, df2])
        elif mode == "test":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "test{}.csv".format(split))
            )
        else:
            assert (
                mode in modes
            ), "You have to choose 'training', 'trainval', 'validation' or 'test' as the dataset mode."
        #    <libs.transformer.ToTensor object at 0x7f3ed3ff3550>和<libs.transformer.TempDownSamp object at 0x7f3ed402abb0>
        self.transform = transform #to tensor and sample
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.augmentations = augmentations
        
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        feature_path = self.dataset_dir + self.df.iloc[idx]["feature"]
        label_path = self.dataset_dir + self.df.iloc[idx]["label"]
        boundary_path = self.dataset_dir + self.df.iloc[idx]["boundary"]

        feature = np.load(feature_path, allow_pickle=True).astype(np.float32) #（C，24000,19,M）

        if (self.dataset == 'MCFS-22') or (self.dataset == 'MCFS-130'):
            feature = feature[:,:,:2] # t,v,c
            feature[:,:,0] = feature[:,:,0]/1280 - 0.5
            feature[:,:,1] = feature[:,:,1]/720 - 0.5
            feature = feature - feature[:,8:9,:]
            feature = feature.transpose(2, 1, 0) #   t,v,c--->c,v,t

        elif (self.dataset == 'PKU-subject') or (self.dataset == 'PKU-view'):
            feature = feature.reshape(-1,2,25,3).transpose(3,0,2,1) #   t,m,v,c--->c,t,v,m
            disps = get_displacements(feature)
            rel_coords = get_relative_coordinates(feature)
            feature = np.concatenate([disps, rel_coords], axis=0)
            feature = feature.transpose(3,0,2,1).reshape(12, 25, -1) #   c,t,v,m--->mc,v,t
        
        elif  (self.dataset == 'LARA'):
            disps = get_displacements(feature)
            rel_coords = get_relative_coordinates(feature)
            feature = np.concatenate([disps, rel_coords], axis=0)
            feature = feature.transpose(3,0,2,1).reshape(12, 19, -1) #   c,t,v,m--->mc,v,t

        elif self.dataset == 'TCG-15':
            feature = feature[:, :, :, np.newaxis]
            feature = feature.transpose(2, 0, 1, 3)
            disps = get_displacements(feature)
            rel_coords = get_relative_coordinates(feature, references=(2))
            feature = np.concatenate([disps, rel_coords], axis=0)
            feature = feature.transpose(3, 0, 2, 1).reshape(6, 17, -1)  # c,t,v,m--->mc,v,t

        label = np.load(label_path).astype(np.int64)
        boundary = np.load(boundary_path).astype(np.float32)
        
            
        if self.transform is not None:
            feature, label, boundary = self.transform([feature, label, boundary])

        if self.augmentations:
            C, V, T = feature.shape

            random_index = torch.rand(1).item()

            # 关节遮挡
            if random_index < 0.35:
                masked_data = feature.clone()
                # Randomly determine the number of joints to mask
                num_masked_joints = torch.randint(1, int(V * 0.5) + 1, (1,)).item()
                # Randomly select the indices of the joints to mask
                masked_joint_indices = torch.randperm(V)[:num_masked_joints]
                # Mask the selected joints across all frames and channels
                masked_data[:, masked_joint_indices, :] = 0
                feature = masked_data

            # rotate
            if random_index > 0.65:
                if (self.dataset == 'MCFS-22') or (self.dataset == 'MCFS-130'):
                    feature = feature
                else:
                    feature = random_rot_3axis(feature, self.dataset)


        sample = {
            "feature": feature,
            "label": label,
            "feature_path": feature_path,
            "boundary": boundary,
        }

        return sample

def random_rot_3axis(data_torch, dataset, theta= math.pi):
    """
    data_numpy: C,T,V,M
    """
    C, V, T = data_torch.shape #torch.Size([3, 64, 25, 2])
    data_torch = data_torch.contiguous().view(-1, 3, V, T)  # T,3,V*M
    # rot = torch.zeros(3).uniform_(-theta, theta)
    if dataset == "LARA" or dataset == "TCG-15":
        rot = torch.zeros(3)
        rot[2] = torch.FloatTensor(1).uniform_(-theta, theta)
        # rot[:2] = torch.FloatTensor(2).uniform_(-0.3, 0.3)
    elif (dataset == 'PKU-subject') or (dataset == 'PKU-view'):
        rot = torch.zeros(3)
        rot[1] = torch.FloatTensor(1).uniform_(-theta, theta)
        # rot[0] = torch.FloatTensor(1).uniform_(-0.3, 0.3)
        # rot[2] = torch.FloatTensor(1).uniform_(-0.3, 0.3)
    rot = _rot_3axis(rot)  # T,3,3
    data_torch = torch.einsum('dc,ncvt->ndvt', rot, data_torch)
    data_torch = data_torch.contiguous().view(C, V, T) #（3,64,25,2）

    return data_torch

def random_rot_2axis(data_torch, theta=0.3):
    """
    data_numpy: C,T,V,M
    """
    C, V, T = data_torch.shape #torch.Size([3, 64, 25, 2])
    data_torch = data_torch.contiguous().view(-1, 2, V, T)  # T,3,V*M
    rot = torch.zeros(1).uniform_(-theta, theta)
    rot = _rot_2axis(rot)  # T,3,3
    data_torch = torch.einsum('dc,ncvt->ndvt', rot, data_torch)
    data_torch = data_torch.contiguous().view(C, V, T) #（3,64,25,2）

    return data_torch

def _rot_3axis(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(1)  # T,1
    ones = torch.ones(1)  # T,1
    # x axis
    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[0:1], -sin_r[0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, sin_r[0:1], cos_r[0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 0)  # T,3,3
    # y axis
    ry1 = torch.stack((cos_r[1:2], zeros, sin_r[1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((-sin_r[1:2], zeros, cos_r[1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 0)
    # z axis
    rz1 = torch.stack((cos_r[2:3], -sin_r[2:3], zeros), dim =-1)
    rz2 = torch.stack((sin_r[2:3], cos_r[2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim = 0)

    rot = rz.matmul(ry).matmul(rx) #row，yaw,pitch
    return rot

def _rot_2axis(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    #
    rx1 = torch.stack((cos_r, -sin_r), dim = -1)  # T,1,3
    rx2 = torch.stack((sin_r, cos_r), dim = -1)  # T,1,3
    rx = torch.cat((rx1, rx2), dim = 0)  # T,3,3

    return rx


def collate_fn(sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_length = max([s["feature"].shape[2] for s in sample])

    feat_list = []
    label_list = []
    path_list = []
    boundary_list = []
    length_list = []

    for s in sample:
        feature = s["feature"]
        label = s["label"]
        boundary = s["boundary"]
        feature_path = s["feature_path"]

        _, _, t = feature.shape
        pad_t = max_length - t
        length_list.append(t)

        if pad_t > 0:
            feature = F.pad(
                feature, (0, pad_t), mode='constant', value=0.)
            label = F.pad(label, (0, pad_t), mode='constant', value=255)
            boundary = F.pad(boundary, (0, pad_t), mode='constant', value=0.)

        # reshape boundary (T) => (1, T)
        boundary = boundary.unsqueeze(0)

        feat_list.append(feature)
        label_list.append(label)
        path_list.append(feature_path)
        boundary_list.append(boundary)

    # merge features from tuple of 2D tensor to 3D tensor
    features = torch.stack(feat_list, dim=0) #（N，C，V，T）
    # merge labels from tuple of 1D tensor to 2D tensor
    labels = torch.stack(label_list, dim=0) #（N，T）

    # merge labels from tuple of 2D tensor to 3D tensor
    # shape (N, 1, T)
    boundaries = torch.stack(boundary_list, dim=0) # (N, 1, T)

    # generate masks which shows valid length for each video (N, 1, T)
    masks = [
        [[1 if i < length else 0 for i in range(max_length)]] for length in length_list
    ]
    masks = torch.tensor(masks, dtype=torch.bool)

    return {
        "feature": features,
        "label": labels,
        "boundary": boundaries,
        "feature_path": path_list,
        "mask": masks,
    }

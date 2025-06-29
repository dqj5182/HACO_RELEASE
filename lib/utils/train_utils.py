import os
import random
import numpy as np

import torch
import torchvision.transforms as transforms


def worker_init_fn(worder_id):
    np.random.seed(np.random.get_state()[1][0] + worder_id)


def get_optim_groups(module):
    """
    Creates parameter groups for a module with specific learning rates and betas,
    excluding parameters that should not be trained (e.g., frozen backbone).

    Args:
        module (nn.Module): The HACO model component.

    Returns:
        List[Dict]: Optimizer parameter groups for HACO.
    """
    trainable_params = {pn: p for pn, p in module.named_parameters() if p.requires_grad}
    frozen_params = {pn: p for pn, p in module.named_parameters() if not p.requires_grad}

    if not trainable_params:
        print("Warning: No trainable parameters found in Module!")

    if frozen_params:
        print(f"Info: {len(frozen_params)} parameters are frozen and will not be updated.")

    # Only include parameters that are trainable in the optimizer groups
    optim_groups = [
        {
            "params": list(trainable_params.values()),
            # Optionally set different learning rates or other optimizer settings
            # "lr": custom_lr,
            # "betas": custom_betas,
        }
    ]
    return optim_groups


def get_transform(backbone_type):
    if 'hamer' in backbone_type:
        transform = transforms.ToTensor()
    elif 'handoccnet' in backbone_type:
        transform = transforms.ToTensor()
    elif 'vit' in backbone_type:
        transform = transforms.ToTensor()
    elif 'resnet' in backbone_type: # follow Hand4Whole
        transform = transforms.ToTensor()
    elif 'hrnet' in backbone_type: # follow METRO
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                    ])
    else:
        raise NotImplementedError

    return transform


def balanced_data(difficulty_factors, aligned_data, target_ratio=1.0):
    """
    Create a balanced dataset from a sorted difficulty list and re-align associated data.

    Args:
        difficulty_factors (np.ndarray): Sorted array of difficulty values (high -> low).
        aligned_data (np.ndarray): Array of data aligned with difficulty_factors.
        target_ratio (float): Proportion of the average fold size to use as the balancing target.

    Returns:
        np.ndarray: Balanced dataset combining all folds.
        np.ndarray: Re-aligned data corresponding to the balanced difficulty list.
    """
    # Ensure input is numpy array for proper indexing
    difficulty_factors = np.array(difficulty_factors, dtype=np.float32)
    aligned_data = np.array(aligned_data)

    # Split based on value range (min to max)
    min_val, max_val = np.min(difficulty_factors), np.max(difficulty_factors)
    # bins = np.linspace(min_val, max_val, 11)

    # exp
    exp_space = np.linspace(1e-5, 1, 6)  # Small epsilon to avoid zero
    beta = 5  # Adjust for desired curvature
    bins = min_val + (max_val - min_val) * np.log1p(exp_space * beta) / np.log1p(beta)

    # # 1.5
    # lin_space = np.linspace(0, 1, 11)
    # bins = min_val + (max_val - min_val) * (lin_space**1.5)  # Slightly denser near max_val

    folds = [
        difficulty_factors[(difficulty_factors >= bins[i]) & (difficulty_factors < bins[i + 1])]
        for i in range(5)
    ]

    aligned_folds = [
        aligned_data[(difficulty_factors >= bins[i]) & (difficulty_factors < bins[i + 1])]
        for i in range(5)
    ]

    # Identify target size for balancing
    fold_sizes = [len(fold) for fold in folds]
    target_size = int(np.mean(fold_sizes) * target_ratio)

    # Balanced sampling logic
    balanced_folds = []
    balanced_aligned_folds = []
    for fold, aligned_fold in zip(folds, aligned_folds):
        if len(fold) > target_size:
            selected_indices = np.random.choice(len(fold), size=target_size, replace=False)
        elif len(fold) < target_size:
            selected_indices = np.random.choice(len(fold), size=target_size, replace=True)
        else:
            selected_indices = np.arange(len(fold))  # Already balanced

        balanced_folds.append(np.array(fold)[selected_indices])
        balanced_aligned_folds.append(np.array(aligned_fold)[selected_indices])

    # Combine balanced folds into one dataset
    balanced_data_out = np.concatenate(balanced_folds)
    balanced_aligned_data_out = np.concatenate(balanced_aligned_folds)

    # Shuffle in unison
    indices = np.arange(len(balanced_data_out))
    np.random.shuffle(indices)

    return balanced_data_out[indices], balanced_aligned_data_out[indices]


def get_contact_difficulty_sample_id(contact_data_path, contact_means_path):
    contact_means = np.load(contact_means_path)

    sample_id_difficulty_list = []
    contact_difficulty_list = []

    for each_file in os.listdir(contact_data_path):
        sample_id = each_file.split('.npy')[0]
        each_contact_data_path = os.path.join(contact_data_path, each_file)
        each_contact_data = np.load(each_contact_data_path)

        if each_contact_data.sum() == 0:
            continue

        contact_difficulty = np.mean(np.abs(each_contact_data) * (1 - contact_means)) - np.mean(np.abs(each_contact_data) * contact_means)

        sample_id_difficulty_list.append(sample_id)
        contact_difficulty_list.append(contact_difficulty)

    contact_diff_pairs = sorted(zip(sample_id_difficulty_list, contact_difficulty_list), key=lambda x: x[1], reverse=True)
    sample_id_difficulty_list, contact_difficulty_list = zip(*contact_diff_pairs)
    sample_id_difficulty_list, contact_difficulty_list = list(sample_id_difficulty_list), list(contact_difficulty_list)

    # Do 10-fold balancing technique
    contact_difficulty_list, sample_id_difficulty_list = balanced_data(contact_difficulty_list, sample_id_difficulty_list)

    return sample_id_difficulty_list
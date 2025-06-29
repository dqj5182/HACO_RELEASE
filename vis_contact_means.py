import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as transforms

from lib.core.config import cfg
from lib.utils.vis_utils import ContactHeatmapRenderer
contact_heatmap_renderer = ContactHeatmapRenderer()


total_contact_data = []
dataset_name_list = cfg.DATASET.train_name

# Get mean of each dataset
for dataset_name in dataset_name_list:
    print(f'Dataset: {dataset_name}')
    contact_data_root_path = f'data/base_data/contact_data/{dataset_name.lower()}'
    save_contact_means_path = os.path.join(contact_data_root_path, f'contact_means_{dataset_name.lower()}.npy')
    save_contact_data_path = os.path.join(contact_data_root_path, f'contact_data_{dataset_name.lower()}.npy')

    os.makedirs(contact_data_root_path, exist_ok=True)

    if os.path.exists(save_contact_means_path) and os.path.exists(save_contact_data_path):
        contact_means = np.load(save_contact_means_path)
        contact_data = np.load(save_contact_data_path)
    else:
        import pdb; pdb.set_trace()
        contact_data_path = f'data/{dataset_name}/preprocessed_data/train/contact_data'
        contact_data = []
        for each_data in tqdm(os.listdir(contact_data_path)):
            each_data_path = os.path.join(contact_data_path, each_data)
            contact = np.load(each_data_path)

            contact_data.append(contact.tolist())
        
        contact_data = np.array(contact_data)
        contact_means = np.mean(contact_data, axis=0)

        if True:
            np.save(save_contact_means_path, contact_means)
            np.save(save_contact_data_path, contact_data)

    total_contact_data.append(contact_data)


# Mean of multiple datasets
try:
    total_contact_data = np.concatenate(total_contact_data, axis=0)
    total_contact_means = np.mean(total_contact_data, axis=0)
except:
    import pdb; pdb.set_trace()


if True:
    total_contact_data = np.array(total_contact_data, dtype=int)
    np.save('data/base_data/contact_data/all/contact_data_all.npy', total_contact_data)
    np.save('data/base_data/contact_data/all/contact_means_all.npy', total_contact_means)



# Visualization
contact_means_render, contact_means_render_list = contact_heatmap_renderer.render_contact(total_contact_means)
cv2.imwrite('contact_means_render_overall.png', contact_means_render)

# for idx in range(len(contact_means_render_list)):
#     cv2.imwrite(f'contact_means_render{idx}.png', contact_means_render_list[idx])
import pdb; pdb.set_trace()
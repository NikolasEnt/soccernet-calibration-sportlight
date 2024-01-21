import os
from typing import List, Tuple, Callable, Optional

import cv2
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader, ConcatDataset, default_collate
from torch.utils.data.dataset import Dataset

from src.datatools.reader import read_json, decode_annot
from src.datatools.intersections import get_intersections
from baseline.evaluate_extremities import scale_points

collate_objs = ['keypoints', 'image', 'img_idx', 'mask']


def custom_collate(batch):
    default_collated = default_collate([{k: v for k, v in sample.items()
                                         if k in collate_objs}
                                        for sample in batch])
    custom_collated = {'raw_annot': [sample['raw_annot'] for sample in batch],
                       'img_name': [sample['img_name'] for sample in batch]}

    return {**default_collated, **custom_collated}


class HRNetDataset(Dataset):
    def __init__(self, dataset_folder: str, transform: Optional[Callable],
                 num_keypoints: int = 30,
                 img_size: Tuple[int, int] = (960, 540),
                 margin: float = 0.0):
        super().__init__()
        self._dataset_folder = dataset_folder
        self.num_keypoints = num_keypoints
        self._transform = transform
        self.img_size = img_size
        self.margin = margin
        self._img_paths = []
        self._raw_labels = []
        self._img_names = []
        for fname in sorted(os.listdir(dataset_folder)):
            if 'info' not in fname:
                annot_path = os.path.join(dataset_folder, fname)
                if annot_path.endswith('.json'):
                    img_path = annot_path.replace('.json', '.jpg')
                    if os.path.exists(img_path):
                        self._img_names.append(fname.replace('.json', '.jpg'))
                        self._img_paths.append(img_path)
                        annot = read_json(annot_path)
                        self._raw_labels.append(annot)

    def __getitem__(self, idx):
        raw_labels = self._raw_labels[idx]
        image = cv2.imread(self._img_paths[idx], cv2.IMREAD_COLOR)
        sample = {
            'image': image,
            'annot': raw_labels,
            'swapped': False
        }
        if self._transform:
            sample = self._transform(sample)
        sample['raw_annot'] = scale_points(sample['annot'], self.img_size[0],
                                           self.img_size[1])
        keypoints, mask = self._annot2keypoints(sample['annot'])
        sample['keypoints'] = keypoints
        sample['img_idx'] = idx
        sample['mask'] = mask
        sample['img_name'] = self._img_names[idx]
        del sample['annot']
        del sample['swapped']
        return sample

    def _annot2keypoints(self, annot) -> np.ndarray:
        kpts_dict, mask = get_intersections(decode_annot(annot),
                                            margin=self.margin)
        keypoints = np.ones(self.num_keypoints * 3, dtype=np.float32) * -1
        for i in range(self.num_keypoints):
            if kpts_dict[i] is not None:
                keypoints[i * 3] = kpts_dict[i][0]
                keypoints[i * 3 + 1] = kpts_dict[i][1]
                keypoints[i * 3 + 2] = 1
            else:
                keypoints[i * 3 + 2] = 0
        mask_vector = np.ones(self.num_keypoints+1, dtype=int)
        for i in mask:
            mask_vector[i] = 0
        return keypoints, mask_vector

    def __len__(self):
        return len(self._raw_labels)


def get_loader(dataset_paths: List[str], data_params: DictConfig,
               transform: Optional[Callable] = None, shuffle: bool = False)\
        -> DataLoader:
    datasets = []
    for dataset_path in dataset_paths:
        datasets.append(HRNetDataset(dataset_path, transform=transform,
                                     num_keypoints=data_params.num_keypoints,
                                     margin=data_params.margin))
    dataset = ConcatDataset(datasets)
    factor = 1 if shuffle else 2
    loader = DataLoader(
        dataset, batch_size=data_params.batch_size * factor,
        num_workers=data_params.num_workers,
        pin_memory=data_params.pin_memory,
        shuffle=shuffle,
        collate_fn=custom_collate)
    return loader

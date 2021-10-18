import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import get_markposion_fromtxt


class medical_dataset(Dataset):
    def __init__(self, img_dir, gt_dir, resize_height, resize_width, point_num, sigma, transform=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.img_names = os.listdir(img_dir)
        self.img_nums = len(self.img_names)
        self.point_num = point_num
        self.sigma = sigma
        self.heatmap_height = int(self.resize_height)
        self.heatmap_width = int(self.resize_width)

    def __getitem__(self, i):
        index = i % self.img_nums
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        img, scal_ratio_w, scal_ratio_h = self.img_preproccess(img_path)
        # img = normalize_robust(img)
        gt_path = self.gt_dir + '/' + img_name.split('.')[0] + '.txt'
        gt_x, gt_y = get_markposion_fromtxt(self.point_num, gt_path)
        x_all = gt_x / scal_ratio_w
        y_all = gt_y / scal_ratio_h
        heatmaps = self.get_heatmaps(x_all, y_all, self.sigma)
        heatmaps_refine = self.get_refine_heatmaps(x_all / 2, y_all / 2, self.sigma)
        # img = self.data_preproccess(img)
        heatmaps = self.data_preproccess(heatmaps)
        heatmaps_refine = self.data_preproccess(heatmaps_refine)
        return img, heatmaps, heatmaps_refine, img_name, x_all, y_all

    def __len__(self):
        return self.img_nums

    def get_heatmaps(self, x_all, y_all, sigma):
        heatmaps = np.zeros((self.point_num, self.heatmap_height, self.heatmap_width))
        for i in range(self.point_num):
            heatmaps[i] = CenterLabelHeatMap(self.heatmap_width, self.heatmap_height, x_all[i], y_all[i], sigma)
        heatmaps = np.asarray(heatmaps, dtype="float32")
        return heatmaps

    def get_refine_heatmaps(self, x_all, y_all, sigma):
        heatmaps = np.zeros((self.point_num, int(self.heatmap_height / 2), int(self.heatmap_width / 2)))
        for i in range(self.point_num):
            heatmaps[i] = CenterLabelHeatMap(int(self.heatmap_width / 2), int(self.heatmap_height / 2), x_all[i],
                                             y_all[i], sigma)
        heatmaps = np.asarray(heatmaps, dtype="float32")
        return heatmaps

    def img_preproccess(self, img_path):
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        img = cv2.resize(img, (self.resize_width, self.resize_height))
        img = np.transpose(img, (2, 0, 1))
        scal_ratio_w = img_w / self.resize_width
        scal_ratio_h = img_h / self.resize_height

        img = torch.from_numpy(img).float()
        # img = img / 255

        # img transform
        transform1 = transforms.Compose([
            # transforms.Normalize([121.78, 121.78, 121.78], [74.36, 74.36, 74.36])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        )
        img = transform1(img)

        return img, scal_ratio_w, scal_ratio_h

    def data_preproccess(self, data):
        data = torch.from_numpy(data).float()
        return data


def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    # heatmap[int(c_y)][int(c_x)] = 2
    return heatmap

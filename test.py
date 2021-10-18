import cv2
import numpy as np
import torch
import xlwt

from config import Config
from utils import get_markposion_fromtxt, get_prepoint_from_htmp

Config = Config()


def get_errors(model, test_loader, note_gt_dir, save_path):
    loss = np.zeros(19)
    num_err_below_20 = np.zeros(19)
    num_err_below_25 = np.zeros(19)
    num_err_below_30 = np.zeros(19)
    num_err_below_40 = np.zeros(19)
    img_num = 0
    for img_num, (img, heatmaps, _, img_name, _, _) in enumerate(test_loader):
        print('图片', img_name[0])
        img = img.cuda(Config.GPU)
        outputs, _ = model(img)
        outputs = outputs[0].cpu().detach().numpy()
        pred = get_prepoint_from_htmp(outputs, Config.scal_w, Config.scal_h)
        note_gt_road = note_gt_dir + '/' + img_name[0].split('.')[0] + '.txt'
        gt_x, gt_y = get_markposion_fromtxt(19, note_gt_road)
        gt_x = np.trunc(np.reshape(gt_x, (19, 1)))
        gt_y = np.trunc(np.reshape(gt_y, (19, 1)))
        gt = np.concatenate((gt_x, gt_y), 1)
        for j in range(19):
            error = np.sqrt((gt[j][0] - pred[j][0]) ** 2 + (gt[j][1] - pred[j][1]) ** 2)
            loss[j] += error
            if error <= 20:
                num_err_below_20[j] += 1
            elif error <= 25:
                num_err_below_25[j] += 1
            elif error <= 30:
                num_err_below_30[j] += 1
            elif error <= 40:
                num_err_below_40[j] += 1

    loss = loss / (img_num + 1)
    num_err_below_25 = num_err_below_25 + num_err_below_20
    num_err_below_30 = num_err_below_30 + num_err_below_25
    num_err_below_40 = num_err_below_40 + num_err_below_30

    row0 = ['NO', '<=20', '<=25', '<=30', '<=40', 'mean_err']
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i])
    for i in range(0, 19):
        sheet1.write(i + 1, 0, i + 1)
        sheet1.write(i + 1, 1, num_err_below_20[i] / (img_num + 1))
        sheet1.write(i + 1, 2, num_err_below_25[i] / (img_num + 1))
        sheet1.write(i + 1, 3, num_err_below_30[i] / (img_num + 1))
        sheet1.write(i + 1, 4, num_err_below_40[i] / (img_num + 1))
        sheet1.write(i + 1, 5, loss[i])
    f.save(save_path)


def predict(model, img_path):
    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape
    img_resize = cv2.resize(img, (Config.resize_w, Config.resize_h))
    img_data = np.transpose(img_resize, (2, 0, 1))
    img_data = np.reshape(img_data, (1, 3, Config.resize_h, Config.resize_w))
    img_data = torch.from_numpy(img_data).float()
    scal_ratio_w = img_w / Config.resize_w
    scal_ratio_h = img_h / Config.resize_h
    img_data = img_data.cuda(Config.GPU)
    outputs = model(img_data)
    outputs = outputs[0].cpu().detach().numpy()
    pred = get_prepoint_from_htmp(outputs, scal_ratio_w, scal_ratio_h)
    return pred

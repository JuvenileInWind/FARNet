import numpy as np


def get_markposion_fromtxt(point_num, path):
    flag = 0
    x_pos = []
    y_pos = []
    with open(path) as note:
        for line in note:
            if flag >= point_num:
                break
            else:
                flag += 1
                x, y = [float(i) for i in line.split(',')]
                x_pos.append(x)
                y_pos.append(y)
        x_pos = np.array(x_pos)
        y_pos = np.array(y_pos)
    return x_pos, y_pos


def get_prepoint_from_htmp(heatmaps, scal_ratio_w, scal_ratio_h):
    pred = np.zeros((19, 2))
    for i in range(19):
        heatmap = heatmaps[i]
        pre_y, pre_x = np.where(heatmap == np.max(heatmap))
        pred[i][1] = pre_y[0] * scal_ratio_h
        pred[i][0] = pre_x[0] * scal_ratio_w
    return pred

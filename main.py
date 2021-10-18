import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from data import medical_dataset
from model import Farnet
from test import get_errors
from train import train_model

model = Farnet()
model.cuda(Config.GPU)

train_set = medical_dataset(Config.img_dir, Config.gt_dir, Config.resize_h, Config.resize_w, Config.point_num,
                            Config.sigma)
test_set1 = medical_dataset(Config.test_img_dir1, Config.test_gt_dir1, Config.resize_h, Config.resize_w,
                            Config.point_num, Config.sigma)
test_set2 = medical_dataset(Config.test_img_dir2, Config.test_gt_dir2, Config.resize_h, Config.resize_w,
                            Config.point_num, Config.sigma)
train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_set1, batch_size=1, shuffle=False, num_workers=4)

criterion = nn.MSELoss(reduction='none')
criterion = criterion.cuda(Config.GPU)
optimizer_ft = optim.Adam(model.parameters(), lr=Config.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_ft, [200], gamma=0.1, last_epoch=-1)
model_ft = train_model(model, criterion, optimizer_ft, scheduler, train_loader, Config.num_epochs)
torch.save(model_ft, Config.save_model_path)

get_errors(model, test_loader, Config.test_gt_dir1, Config.save_results_path)

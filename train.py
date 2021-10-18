import torch

from config import Config

Config = Config()


def train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs=25):
    for epoch in range(0, num_epochs):
        print('Epoch{}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()
        loss_temp = 0
        for i, (img, heatmaps, heatmaps_refine, img_name, x_all, y_all) in enumerate(train_loader):
            img = img.cuda(Config.GPU)
            heatmaps = heatmaps.cuda(Config.GPU)
            heatmaps_refine = heatmaps_refine.cuda(Config.GPU)
            outputs, outputs_refine = model(img)

            loss = criterion(outputs, heatmaps)
            ratio = torch.pow(Config.base_number, heatmaps)
            loss = torch.mul(loss, ratio)
            loss = torch.mean(loss)
            loss_temp += loss

            loss_refine = criterion(outputs_refine, heatmaps_refine)
            ratio_refine = torch.pow(Config.base_number, heatmaps_refine)
            loss_refine = torch.mul(loss_refine, ratio_refine)
            loss_refine = torch.mean(loss_refine)

            loss = loss + loss_refine
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model

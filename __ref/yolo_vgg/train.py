import os
import torch
from config import device
import time


def train(epoch, vis, train_loader, model, criterion, optimizer, scheduler, opts):

    tic = time.time()
    model.train()

    # warm up training
    if epoch < 5:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
    elif epoch == 5:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4

    for idx, datas in enumerate(train_loader):
        images = datas[0]
        boxes = datas[1]
        labels = datas[2]

        # assign to cuda
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # feed forward
        preds = model(images)  # B, 425, 13, 13
        preds = preds.permute(0, 2, 3, 1)  # B, 13, 13, 425

        # loss
        loss, losses = criterion(preds, boxes, labels)

        # backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        toc = time.time()

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # for each steps
        if idx % 100 == 0:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'Learning rate: {lr:.7f} s \t'
                  'Training Time : {time:.4f}\t'
                  .format(epoch, idx, len(train_loader),
                          loss=loss,
                          lr=lr,
                          time=toc - tic))

            if vis is not None:
                vis.line(X=torch.ones((1, 6)).cpu() * idx + epoch * train_loader.__len__(),  # step
                         Y=torch.Tensor([loss, losses[0], losses[1], losses[2], losses[3], losses[4]]).unsqueeze(
                             0).cpu(),
                         win='train_loss',
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='training loss',
                                   legend=['Total Loss', 'xy_loss', 'wh_loss', 'conf_loss', 'no_conf_loss',
                                           'cls_loss']))

    if not os.path.exists(opts.save_path):
        os.mkdir(opts.save_path)
    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    if scheduler is not None:
        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict()}

    torch.save(checkpoint, os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'.format(epoch))
    print('save of epoch {} parameters...'.format(epoch))

    if scheduler is not None:
        scheduler.step()

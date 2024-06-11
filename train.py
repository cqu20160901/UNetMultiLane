from getdata import GetLaneDataset
from torch.utils.data import DataLoader as DataLoader
from unet import UNetMultiLane as Net
import torch
from torch.autograd import Variable
from loss import softmax_focal_loss
from tqdm import tqdm
import sys


train_image_txt = './data/VIL100/train.txt'
test_image_txt = './data/VIL100/test.txt'

data_main_path = './data/VIL100'
weights_save_path = './weights'

input_height = 480
input_width = 640
lane_num = 9   # 8 条车道线 + 1
type_num = 11  # 10 种线的类型 + 1

epoch_num = 50
batch_size = 2
learn_rate = 0.001
num_workers = 1

width_mult = 0.25


def test_eval(model, epoch):
    log_txt = open(weights_save_path + './log.txt', 'a')
    dataset = GetLaneDataset(image_txt=test_image_txt, data_main_path=data_main_path, input_height=input_height, input_width=input_width, train_mode=False)
    images_num = len(dataset)
    print('eval images num is:', images_num)
    model.eval()
    criterion = softmax_focal_loss

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    seg_loss_total = 0
    cls_loss_total = 0

    for image, label_mask, label_type in tqdm(dataloader):

        image, label_mask, label_type = Variable(image).cuda(), Variable(label_mask).cuda(), Variable(label_type).cuda()
        pred_out = model(image)
        seg_loss, cls_loss = criterion(pred_out, label_mask, label_type.squeeze())

        seg_loss_total += seg_loss.item()
        cls_loss_total += cls_loss.item()

    print(f"{'eval_seg_loss':>13} {'eval_cls_loss':>13} {'eval_total_loss':>15}")
    print(f"{seg_loss_total:>13} {cls_loss_total:>13} {(seg_loss_total + seg_loss_total):>15}")

    save_line = 'epoch:' + str(epoch) + ',seg_loss_total:' + str(seg_loss_total) + ', cls_loss_total:' + str(cls_loss_total) + ', total_loss:' + str(seg_loss_total + seg_loss_total) + '\n'
    log_txt.write(save_line)
    log_txt.close()

    return seg_loss_total + seg_loss_total


def train():
    dataset = GetLaneDataset(image_txt=train_image_txt, data_main_path=data_main_path, input_height=input_height, input_width=input_width, train_mode=True)
    images_num = len(dataset)
    print('train images num is:', images_num)

    model = Net(in_channels=3, lane_num=lane_num, type_num=type_num, width_mult=width_mult, is_deconv=True, is_batchnorm=True, is_ds=True)
    model = model.cuda()
    model.train()

    criterion = softmax_focal_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    print(f"{'epoch'}/{'epoch_num'} | {'seg_loss':>8} | {'cls_loss':>8} | {'total_loss':>10}")

    eval_loss_total = 1e7
    best_epoch = 0
    for epoch in range(epoch_num):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

        seg_loss_total = 0
        cls_loss_total = 0

        for image, label_mask, label_type in tqdm(dataloader):

            image, label_mask, label_type = Variable(image).cuda(), Variable(label_mask).cuda(), Variable(label_type).cuda()
            pred_out = model(image)
            seg_loss, cls_loss = criterion(pred_out, label_mask, label_type.squeeze())
            total_loss = seg_loss + cls_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            seg_loss1 = "%.4f" % seg_loss.item()
            cls_loss1 = "%.4f" % cls_loss.item()
            total_loss1 = "%.4f" % total_loss.item()

            text = f"{epoch}/{epoch_num} {seg_loss1:>8} {cls_loss1:>8} {total_loss1:>8}"
            sys.stdout.write(text)
            sys.stdout.flush()

            seg_loss_total += seg_loss.item()
            cls_loss_total += cls_loss.item()

        seg_loss_total1 = "%.4f" % seg_loss_total
        cls_loss_total1 = "%.4f" % cls_loss_total
        total_loss_total1 = "%.4f" % (seg_loss_total + cls_loss_total)

        print()
        print(f"{'epoch':<5} {'epoch_num':<9} {'seg_loss_total':>14} {'cls_loss_total':>14} {'total_loss_total':>16}")
        print(f"{epoch:<5} {epoch_num:<9} {seg_loss_total1:>14} {cls_loss_total1:>14} {total_loss_total1:>16}")

        torch.save(model.state_dict(), weights_save_path + '/epoch_{0}.pth'.format(epoch + 1))

        eval_loss = test_eval(model, epoch)
        model.train()
        if eval_loss < eval_loss_total:
            eval_loss_total = eval_loss
            torch.save(model.state_dict(), weights_save_path + '/best.pth')
            best_epoch = epoch
    log_txt = open(weights_save_path + './log.txt', 'a')
    save_line = 'eval best epoch is:' + str(best_epoch) + '\n'
    log_txt.write(save_line)
    log_txt.close()


if __name__ == '__main__':
    print('start train ...')
    train()


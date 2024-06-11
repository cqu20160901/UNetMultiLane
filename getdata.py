import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


def train_data_transform(input_height, input_width):
    train_image_transform = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor()
    ])
    return train_image_transform


def eval_data_transform(input_height, input_width):
    eval_image_transform = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor()
    ])
    return eval_image_transform


def get_images_labels(data_main_path, labels_path):
    with open(labels_path, "r") as f:
        lines = f.readlines()
    images_list = []
    labels_list = []
    label_type_list = []
    for line in lines:

        name = line.split('\n')[0].split(' ')

        image_path = data_main_path + name[0]
        images_list.append(image_path)

        label_path = data_main_path + name[1]
        labels_list.append(label_path)

        label_type = name[10:18]
        label_type = np.asarray(label_type).astype('int64')
        for i in range(len(label_type)):
            # 可能是有些线的类别超出了给定的10中线的类型（定义的是10种线类型，标签中有些超过了10）
            if label_type[i] >= 11:
                label_type[i] = 0

        label_type_list.append(label_type)

    return images_list, labels_list, label_type_list


class GetLaneDataset(Dataset):
    def __init__(self, image_txt, data_main_path, input_height, input_width, train_mode):
        self.images_path, self.labels_path, self.label_types = get_images_labels(data_main_path, image_txt)
        self.train_mode = train_mode
        self.input_height = input_height
        self.input_width = input_width
        if train_mode:
            self.data_transform = train_data_transform(input_height, input_width)
        else:
            self.data_transform = eval_data_transform(input_height, input_width)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):

        image = Image.open(self.images_path[idx])

        label_seg = Image.open(self.labels_path[idx])
        label_mask = F.resize(label_seg, [self.input_height, self.input_width], Image.NEAREST)
        label_mask = np.array(label_mask)

        label_type = self.label_types[idx]

        if self.train_mode:
            return self.data_transform(image), torch.LongTensor([label_mask]), torch.LongTensor([label_type])
        else:
            return self.data_transform(image), torch.LongTensor([label_mask]), torch.LongTensor([label_type])


if __name__ == '__main__':
    print('This is main ....')
    image_txt = r'./VIL100/train.txt'
    data_main_path = r'./VIL100'

    train_set = GetLaneDataset(image_txt=image_txt, data_main_path=data_main_path, input_height=256, input_width=256, train_mode=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

    for i, data in enumerate(train_loader, 0):
        inputs, labels, labels_type = data
        print(inputs.shape)
        print(labels.shape)
        print(labels_type.shape)
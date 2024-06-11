from unet import UNetMultiLane as Net
import torch
import numpy as np
import cv2
from torch.autograd import Variable


width_mult = 0.25

input_height = 480
input_width = 640
lane_num = 9   # 8 条车道线 + 1
type_num = 11  # 10 种线的类型 + 1


color_list = [(100, 149, 237), (0, 0, 255), (173, 255, 47), (240, 255, 255), (0, 100, 0),
              (47, 79, 79), (255, 228, 196), (138, 43, 226), (165, 42, 42), (222, 184, 135)]

lane_Id_type = [7, 5, 3, 1, 2, 4, 6, 8]


line_type = ['No lane markings',
             'Single white solid line',
             'Single white dashed line',
             'Single solid yellow line',
             'Single yellow dashed line',
             'Double solid white lines',
             'Double solid yellow lines',
             'Double yellow dashed lines',
             'Double white yellow solid lines',
             'Double white dashed lines',
             'Double white solid dashed lines']


def precess_image(img_src, resize_w, resize_h):
    image = cv2.resize(img_src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image /= 255
    image = image.transpose((2, 0, 1))
    return image


def softmax(x, axis):
    x -= np.max(x, axis=axis, keepdims=True)
    value = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return value


def test_image():
    weights_path = './weights/best.pth'
    image_path = './images/test.jpg'

    model = Net(in_channels=3, lane_num=lane_num, type_num=type_num, width_mult=width_mult, is_deconv=True, is_batchnorm=True, is_ds=True)
    model = model.cuda()
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    origin_image = cv2.imread(image_path)
    image_height, image_width = origin_image.shape[:2]

    input_image = precess_image(origin_image, input_width, input_height)

    input_image = torch.from_numpy(input_image)
    input_image = Variable(input_image.unsqueeze(0)).cuda()
    output = model(input_image)

    seg_output = softmax(output[0].cpu().detach().numpy(), axis=1)[0]
    cls_output = softmax(output[1].cpu().detach().numpy(), axis=2)[0]

    cls_output = np.argmax(cls_output, axis=1)

    mask = np.zeros(shape=(input_height, input_width, 3))

    lane_id = []
    write_pos = []

    for i in range(mask.shape[0] - 1, 0, -1):
        for j in range(mask.shape[1] - 1, 0, -1):
            max_index = np.argmax(seg_output[:, i, j])
            if max_index not in lane_id:
                lane_id.append(max_index)
                if i > input_height - 20 or j > input_width - 20:
                    write_pos.append([j - 20, i - 20])
                else:
                    write_pos.append([j, i])
            if max_index != 0 and seg_output[max_index, i, j] > 0.5:
                mask[i, j, :] = color_list[max_index]

    mask = cv2.resize(mask, (image_width, image_height))

    for i in range(len(lane_id)):
        if lane_id[i] == 0:
            continue

        lane_type = cls_output[lane_Id_type.index(lane_id[i])]

        px = int(write_pos[i][0] / input_width * image_width)
        py = int(write_pos[i][1] / input_height * image_height)

        cv2.putText(origin_image, str(lane_id[i]), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(origin_image, str(lane_type), (px, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(origin_image, 'lane_id: 7-5-3-1-2-4-6-8', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(origin_image, 'line type:', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    for i in range(len(line_type)):
        cv2.putText(origin_image, str(i) + ': ' + str(line_type[i]), (10, 80 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 0), 2, cv2.LINE_AA)

    opencv_image = np.clip(np.array(origin_image) + np.array(mask) * 0.4, a_min=0, a_max=255)
    opencv_image = opencv_image.astype("uint8")
    cv2.imwrite('./test_pytorch_result.jpg', opencv_image)


if __name__ == '__main__':
    print('test image ...')
    test_image()


# UNetMultiLane
UNetMultiLane 多车道线、车道线类型识别，自我学习使用，没有经过大量测试，不免有问题，不喜不扰。

# 模型结构说明

基于UNet 分割模型增加了检测头来识别车道线的类型（单实线、双黄线等），可以识别出"所在车道"和"车道线类型"。


# 数据说明

基于开源数据集 VIL100。其中数据标注了所在的六个车道的车道线和车道线的类型。

8条车道线（六个车道），对应的顺序是：7,5,3,1,2,4,6,8。其中1,2对应的自车所在的车道，从左往右标记。

车道线的类别（10个类别）：单条白色实线、单条白色虚线、单条黄色实线、单条黄色虚线、双条白色实线、双条黄色实线、双条黄色虚线、双条白色实虚线、双条白色黄色实线、双条白色虚实线。

**本仓库提供了20张图像和标签，可以直接训练起来。**


说明示例：

train.txt

/JPEGImages/0_Road001_Trim003_frames/00045.jpg /Annotations/0_Road001_Trim003_frames/00045.png 0 0 0 1 1 1 1 1 0 0 0 1 2 10 2 1

![00045](https://github.com/cqu20160901/UNetMultiLane/assets/22290931/373173e9-ff33-41f4-a4f9-94ee7b071691)

标签说明：
/JPEGImages/0_Road001_Trim003_frames/00045.jpg   图片
/Annotations/0_Road001_Trim003_frames/00045.png  分割标签（png的p模式存储的标签） 

0 0 0 1 1 1 1 1 0 0 0 1 2 10 2 1 共十六个值，前8个是对应的车道（8条车道线，对应的顺序是7,5,3,1,2,4,6,8），后8个值是线的类别（10个类别）

0 0 0 1 1 1  1 1（共8条车道线，左起前三条000无车道线，11111有到车道）

0 0 0 1 2 10 2 1（每条车道线对应的类型，左起前三条000无车道线，1 2 10 2 1分别对应的车道线类型）


# 分割效果

![image](https://github.com/cqu20160901/UNetMultiLane_onnx_tensorRT_rknn_horizon/blob/main/onnx/test_result.jpg)


# 部署参考

[部署](https://github.com/cqu20160901/UNetMultiLane_onnx_tensorRT_rknn_horizon)

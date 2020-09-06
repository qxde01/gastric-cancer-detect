# gastric-cancer-detect
胃癌恶性病变组织检测
# 数据准备
将图片路径及标签放在一个csv文件里，可以用·data.py·生成 格式如下：

|filepath|label|maskpath|type|filename|
|:---:|:---:|:---:|:---:|:---:|
|F:/data/cancer/0/3375.jpg|0|train|3375.jpg|
|F:/data/cancer/0/0425.jpg|0|train|0425.jpg|


# Reference
* [“华录杯”SEED江苏大数据开发与应用大赛——癌症风险智能诊断](https://www.marsbigdata.com/competition/details?id=5815639985152)
* [pytorch-U2Net](https://github.com/NathanUA/U-2-Net),  [paper](https://arxiv.org/pdf/2005.09007v1.pdf)
* [pytorch-Unet3](https://github.com/ZJUGiveLab/UNet-Version),[paper](https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
* [keras-unet](https://github.com/zhixuhao/unet)
* [UEfficientNetB4](https://www.kaggle.com/meaninglesslives/nested-unet-with-efficientnet-encoder)

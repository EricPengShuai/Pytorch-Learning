# 卷积神经网络

## 概述

卷积神经网络——Conventional Neural Network，一般用于图像分类、图像检索、图像语义分割、物体检测等计算机视觉问题

### 前馈运算

输入原始数据（RGB图像、原始音频数据等），经过**卷积、池化、激活函数**等一系列层叠操作，最后输出目标结果（分类、回归等结果）『或者说得到一个目标函数』。

<img src="https://i.loli.net/2021/05/24/gYvtOkECfMA6ebJ.png"  alt="前馈运算" style="zoom:60%;"/>

### 反馈运算

然后计算预测结果和实际结果的Loss，凭借**反向传播算法**将误差由最后一层逐层向前**反馈**，更新每层参数，并在更新参数之后再次前馈，如此往复，直到网络模型收敛。

> 最小化损失函数更新参数：随机梯度下降（Stochastic Gradient Descent，SDG）、误差反向传播
>
> 对于大规模的问题，批处理随机梯度下降（mini-batch SGD），随机选择n个样本处理完成前馈运算和反馈运算就是一个“**批处理过程**”（mini-batch），不同批处理之间无放回的处理完所有样本就是“**一轮**”（epoch），其中批处理样本的大小（batch-size）不宜过小，上限取决于硬件资源。

### 基本术语

- **Channel**：输入数据（图像）个数
- **Convolution Layer**：经过卷积之后得到的中间结果
- **Pooling Layer**：经过池化之后得到的中间结果
- **Flatten**：将每个图像比如6*6，展开成一个vector（size = 36\*1）
- **Padding and Stride**：图像周围补充0，每次filter移动步长，涉及到超参数的调整
- **Softmax**：激活函数$S_i = \frac{e^i}{\Sigma{e^j}}$：将一些输入映射为0-1之间的实数，并且归一化保证和为1
- **Fully Connected Layer**



## 基本操作

输入数据一般是三维张量**tensor**，比如n个video frames（也可以叫做channel），每个frame大小是6\*6，然后经过**filter**（比如是3\*3大小，这参数是通过训练得到的，可以使用比如SGD），在经过**Max/ Average Pooling**（比如大小是2*2），经过多次处理最后**Flatten**成vector，然后输入到一个全连接神经网络中，其中可以使用**softmax**，然后输出结果

<img src="https://i.loli.net/2021/05/24/mBgvl9QX1qFMYSL.png" alt="CNN结构" style="zoom:40%;" />

### 卷积

这个过程就是**特征提取**，通俗点说就是对于冗余信息压缩提纯，进行这个过 程我们需要学习一些参数（这些参数组合成一个**卷积核**，Filter Matrix），然后对输入的图像进行扫描，每次移动步长**stride**，另外如果需要充分利用图像信息也可以**padding**

<img src="https://i.loli.net/2021/05/24/qyXrAnOocJiL3zM.png" alt="convolution" style="zoom:40%;"/>



- 每个filter可以探测到一些小特征，这也是选择其中参数的依据
- 原始图像经过filter之后可以看到一些特征，对应在灰度图中可能是一条斜杠

<img src="https://i.loli.net/2021/05/24/OIV7n4qRw5mJBAr.png" alt="property" style="zoom:50%;"/>

- filter的个数决定了卷积操作输出的层数，所以每次经过卷积之后输出结果一般都会变胖

<img src="https://i.loli.net/2021/05/24/zxnTUCBmHQJLZRV.png" alt="卷积层逐渐变胖"/>

- 卷积过程中的参数是共享的，**Shared Weights**，所以需要的参数就会变少

<img src="https://i.loli.net/2021/05/24/ihmd4xAck5v2rsN.png" alt="Shared Weights" style="zoom:60%;"/>

### 池化

对卷积层输出的特征图**进一步特征提取**，主要分为**Max Pooling**、**Average Pooling**

<img src="https://i.loli.net/2021/05/24/scUzpRSlgkoPXGD.png" alt="Pooling" style="zoom:50%;"/>

> 池化实际上是一种”降采样“（down-sampling）操作。一般有三种功能：
>
> 1. 特征不变性
> 2. 特征降维
> 3. 防止过拟合

### 输出

之前得到的所有特征需要使用Flatten将特征矩阵转化成$vector$，最后将$vector$接到全连接神经网络上，全连接层则起到将学到的特征表示映射到样本的标记空间，使用激活函数**softmax**得到分类或者回归的结果

<img src="https://i.loli.net/2021/05/24/RjbFrXGTOApm5gc.png"  alt="Flatten" style="zoom:50%;"/>



## 参考文献

1. b站 李宏毅 https://www.bilibili.com/video/BV1JE411g7XF?p=17

2. b站   [阿力阿哩哩](https://space.bilibili.com/299585150) https://www.bilibili.com/video/BV1j7411f7Ru?t=665

3. 魏秀参《解析深度学习——卷积神经网络原理与视觉实践》（开源电子版）


# Pytorch-Learning
Pytorch Framework learning for deeplearning  

## CNN
### 基础知识
- 参考 [CNN](./CNN/CNN.md)

### 相关应用
#### MINST Classification
- [CNN_MINST.ipynb](./CNN/CNN_MINST.ipynb): 使用 [torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) 做手写数字书识别  
    - 其中使用的是`./dataset/MINST`文件夹的`names_test.csv.gz`和`names_train.csv.gz`数据集  

## RNN
### 基础知识
- 需要理解输入维度和隐层维度：

  ```python
  # RNN需要指明输入大小、隐层大小以及层数（默认为1）
  cell = torch.nn.RNN(input_size, hidden_size, num_layers)
  
  # input: (seq_len, batch_size, input_size) 所有时间点的第一层状态
  # hidden: (num_layers, batch_size, hidden_size) 第一个时刻所有层的状态
  
  # out: (seq_len, batch_size, hidden_size) 所有时间点的最后一层状态
  # hidden: (num_layers, batch_size, hidden_size) 最后时刻的所以层状态
  out, hidden = cell(inputs, hidden)
  ```

- 自然语言的输入处理需要学会word embedding

- 另外可以参考 [RNN](./RNN/RNN.md)

### 相关应用

[RNNcell.ipynb](./RNN/RNNcell.ipynb): 学习使用 [torch.nn.RNNcell](https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html?highlight=rnncell#torch.nn.RNNCell), 用于`hello --> ohlol`  
    - 其实也是一个分类问题  

#### RNN for Regression  
- [RNN_Regression.py](./RNN/RNN_Regression.py): 使用 [torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) 模拟`sin(x)`逼近`cos(x)`  
   - 效果图：  

![RNN_Regression](https://i.loli.net/2021/03/12/4ozBxbLsX1c6f3J.gif)


#### GRU for Classification

- [GRU_Classifier.ipynb](./RNN/GRU_Classifier.ipynb): 使用 [torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) 训练名字到国家的分类，即输入名字输出其属于哪个国家的
    - 其中使用的数据集在`./dataset`文件夹中  
    - 对应的Python脚本文件可以参考: [GRU_Classifier.py](./RNN/GRU_Classifier.py)

#### LSTM for Regression and Prediction
- [LSTM_Regression.py](./LSTM/LSTM_Regression.py): 使用 [torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) 模拟`sin(x)`逼近`cos(x)`  
   - 效果图:  

![LSTM_Regression](https://i.loli.net/2021/03/12/7OJvI1sP26HuzAF.gif)

- [LSTM_airplane_forcast.ipynb](./LSTM/LSTM_airplane_forcast.ipynb): 根据前9年的数据预测后3年的客流, 这个为了训练过程简单使用的数据是: `./dataset/airplane_data.csv`，只有144条数据，所以训练效果不是那么好，只是为了简单理解LSTM做回归分析的方法

   **- 这个例子好像是有点问题的：根据[简书](https://www.jianshu.com/p/18f397d908be)的评论**

   > 1. Data leakage的问题：数据的预处理放在了数据集分割之前，测试集的信息泄露给了训练集
   > 2. 下面讨论最多的input_size和seq_len的问题：若本例目的是"以t-2,t-1的数据预测t的数据"的话
   >    根据Pytorch的DOC定义“input of shape (seq_len, batch, input_size)”
   >    而本例的输入维度从这可以看出来了train_X = train_X.reshape(-1, 1, 2)
   >    说明input_size=2 batch=1 seq_len=？（我没算），不过这似大概没能用到LSTM的特性，或者说没法用"以t-2,t-1的数据预测t的数据"来解释本结构的目的。
   >    我比较同意上面讨论的人的看法，即特征数（input_size）为1，seq_len为2，应该是比较合理的





## Convolutional LSTM

一般来说CNN可以提取图片的空间特征，LSTM可以提取时间特征，如果有时间序列的图片场景，我们可以使用 **Convolutional LSTM (ConvLSTM)** 提取**时空特征**（Spatio-temporal features），该网络由香港科技大学的 Shi Xingjian 等人提出，具体的论文可以参考：[Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)

![ConvLSTM](https://s2.loli.net/2022/01/11/n7FSVUbqstr28dj.png)

> 代码参考：[ConvLSTM.py](./ConvLSTM.py)，其中主要有四个类：
>
> - `ConvLSTMCell`是卷积LSTM的细胞单元：通过卷积操作计算之后返回LSTM中隐层状态和细胞状态；
> - `ConvLSTM`是实现卷积LSTM提取时空特征的一个模型：输入格式为 $(B, T, C, H, W)$ 或者 $(T, B, C, H, W)$，B是batchSize，T是timeLength或seqLen，CHW分别是图片的channel、height和width
>   - PyTorch中CNN的输入形状为：$(B,C_{in},H,W)$ ，[参考链接](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv#torch.nn.Conv2d)
>   - PyTorch中LSTM的输入形状为：$(B,T,Hidden_{in})$ 或者 $(T,B,Hidden_{in})$ ，[参考链接](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM)
> - `outputCNN`：一个简单的CNN网络处理ConvLSTM的输出
>   - 一般LSTM的输出后面都会接一个Linear层处理，这里ConvLSTM的输出就是使用CNN处理
> - `ConvLSTM_model`：使用ConvLSTM以及outputCNN的混合网络



## Transformer

- 参考 [transformer.md](./Transformer.md)

## 其他  
1. [multiple_dimension_diabetes.ipynb](./LinearNetwork/multiple_dimension_diabetes.ipynb): 学习处理多维特征输入，使用**二分类交叉熵** [torch.nn.BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html?highlight=bce#torch.nn.BCELoss)  
    - 使用的`./dataset`文件夹中的`diabetes.csv.gz`数据集  

2. [softmax_classifier.ipynb](./softmax_classifier.ipynb): 学习处理多维特征分类，使用**交叉熵** [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropy#torch.nn.CrossEntropyLoss)  
    - 使用的是`./dataset/MINST`数据集  


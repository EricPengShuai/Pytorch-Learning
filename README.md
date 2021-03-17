# Pytorch-Learning
Pytorch Framework learning for deeplearning  

## RNN for Regression  
1. `LSTM_Regression.py`: 使用[torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)模拟`sin(x)`逼近`cos(x)`  
   
	- 效果图:  

![LSTM_Regression](https://i.loli.net/2021/03/12/7OJvI1sP26HuzAF.gif)



2. `RNN_Regression.py`: 使用[torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)模拟`sin(x)`逼近`cos(x)`  
   - 效果图：  

![RNN_Regression](https://i.loli.net/2021/03/12/4ozBxbLsX1c6f3J.gif)

  
3. `LSTM_airplane_forcast.ipynb`: 根据前9年的数据预测后3年的客流, 这个为了训练过程简单使用的数据是: `./dataset/airplane_data.csv`，只有144条数据，所以训练效果不是那么好，只是为了简单理解LSTM做回归分析的方法


## CNN and RNN for Classify  
1. `CNN_MINST.ipynb`: 使用[torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)做手写数字书识别  
    - 其中使用的是`./dataset/MINST`文件夹的`names_test.csv.gz`和`names_train.csv.gz`数据集  

2. `GRU_Classifier.ipynb`: 使用[torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)训练名字到国家的分类，即输入名字输出其属于哪个国家的
    - 其中使用的数据集在`./dataset`文件夹中  

## 其他  
1. `RNNcell.ipynb`: 学习使用[torch.nn.RNNcell](https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html?highlight=rnncell#torch.nn.RNNCell), 用于`hello --> ohlol`  
    - 其实也是一个分类问题  

2. `multiple_dimension_diabetes.ipynb`: 学习处理多维特征输入，使用二分类交叉熵[torch.nn.BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html?highlight=bce#torch.nn.BCELoss)  
    - 使用的`./dataset`文件夹中的`diabetes.csv.gz`数据集  

3. `softmax_classifier.ipynb`: 学习处理多维特征分类，使用交叉熵[torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropy#torch.nn.CrossEntropyLoss)  
    - 使用的是`./dataset/MINST`数据集  


# Transformer

<img src="https://z3.ax1x.com/2021/04/20/c7wyKU.png#shadow"  alt="transformer框架图" style="zoom:70%" />

## Encoder

上图左半部分，6层结构，每层结构包含两个子层

### 1. 输入：字向量与位置编码

$$
X_{embedding} = X_{word-embedding} + X_{pos-embedding}
$$



### 2. 多头注意力层

multi-head attention：**(Q、K、V)**
$$
Q = \text{Linear}_q(X) = XW_{Q}\\
K = \text{Linear}_k(X) = XW_{K}\\
V = \text{Linear}_v(X) = XW_{V}\\
X_{attention} = \text{Self-Attention}(Q,K,V)
$$
注意：Ecoder的Q是embedding来的，是已知的，而Decoder输出的Q是预测的，也就是结果预测的词

### 3. self_attention残差连接和Layer Normalization

注意这里的Norm是Layer Norm，而不是BN（效果不太好）
$$
X_{attention} = X + X_{attention}\\
X_{attention} = \text{LayerNorm}(X_{attention})
$$


### 4. 前馈连接层

feed forward：就是全连接。其实就是两层线性映射并用激活函数激活，比如说 ReLU
$$
X_{hidden} = \text{Linear}(\text{ReLU}(\text{Linear}(X_{attention})))
$$

### 5. feed_forward残差连接和Layer Normalization

$$
X_{hidden} = X_{attention} + X_{hidden}\\
X_{hidden} = \text{LayerNorm}(X_{hidden})
$$

其中 $X_{hidden} \in \mathbb{R}^{batch\_size  \ * \  seq\_len. \  * \  embed\_dim}$

<img src="https://z3.ax1x.com/2021/04/20/c7wD2V.png#shadow" alt="Encoder结构" style="zoom:80%;" />

## Decoder

上图右半部分，6层结构，每层结构包含三个子层



### **遮掩多头注意力层**

- mask防止训练过程使用未来输出的单词，**保证预测位置i的信息只能基于比i小的输出**
- encoder层可以并行计算，decoder层像RNN一样一个一个训练，需要使用上一个位置的输入当做attention的query

<img src="https://z3.ax1x.com/2021/04/20/c7w48x.png#shadow" style="zoom:80%;" />



### **多头注意力结构**

- 输入来源1：第一个子层的输出
- 输入来源2：Encoder层的输出

这是和encoder层区别的地方以及这里的信息交换是权值共享



### **前馈连接层**

**残差连接**：Output Embedding -->  Add & Norm， Add & Norm --> Add & Norm， Add & Norm --> Add & Norm

- 残差结构可以解决梯度消失问题，增加模型的复杂性

- Norm层为了对attention层的输出进行分布归一化（对一层进行一次归一化），区别于cv中的batchNorm（对一个batchSize中的样本进行归一化）




## 几个术语

- **self-attention**：是transformer用来找到重点关注与当前单词相关词语的一种方法

- **Multi-Headed Attention**：多头注意力机制是指有多组Q,K,V矩阵，一组Q,K,V矩阵代表一次注意力机制的运算，transformer使用了8组，所以最终得到了8个矩阵，将这8个矩阵拼接起来后再乘以一个参数矩阵WO,即可得出最终的多注意力层的输出。

- **Positional Encoding**：为了解释输入序列中单词顺序而存在，维度和embedding的维度一致。这个向量决定了当前词的位置，或者说是在一个句子中不同的词之间的距离。

- **layer normalization**：在transformer中，每一个子层（自注意力层，全连接层）后都会有一个Layer normalization层



## 参考

1. [b站视频讲解](https://urlify.cn/yiuMNv)-[Transformer](https://wmathor.com/index.php/archives/1438/)的Pytorch实现 
2. [Pytorch代码实现](https://wmathor.com/index.php/archives/1455/)
3. [b站-Transformer从零详细解读](https://urlify.cn/qiqAJf)
4. [知乎-Transformer面试题系列](https://zhuanlan.zhihu.com/p/148656446)
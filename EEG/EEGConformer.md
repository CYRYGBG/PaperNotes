## EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization

### 简介

论文认为，在EEG分类中使用卷积神经网络进行分类，模型只能从脑电图中提取出局部的时间特征，而无法提取出EEG中的前后的长期依赖性。因此，论文提出了EEG Conformer来同时提取局部和全局的特征。

论文总结为三个贡献：

- 提出了EEG Conformer**同时提取局部和全局特征**，在三个公共数据集上取得了SOTA
- 进行了大量实验来研究Transformer模块和Attention模块**受参数变化的影响**
- 提出了一种**新的可视化方法**来研究模型是如何提取全局特征的

### 模型结构

![image-20250325183849483](https://gitee.com/fbanhua/figurebed/raw/master/images/20250325183849534.png)

模型主要由三部分构成：卷积部分、自注意力部分和最后的分类器部分。其中，卷积部分负责提取局部的时间特征和空间特征；自注意力部分负责提取全局的时间特征；分类器部分负责输出最后的分类结果。

![image-20250325184644521](https://gitee.com/fbanhua/figurebed/raw/master/images/20250325184644548.png)

在卷积部分中，模型**接受原始的二维脑电图作为输入**（维度表示为[1, channel, sample]，1维度是为了与后面的卷积结果中的多张特征图进行统一），第一个时间卷积核（kernel size(1, 25)）在每个通道的时间维度上提取**时间特征**，第二个卷积核（kernel size(ch, 1)）提取每个时间点上不同通道之间的**空间特征**，最后使用平均池化来**抑制噪声的干扰**。

在自注意力部分中，模型以卷积部分的结果按时间维度划分，所有通道的信息聚合为一个token，然后在这些token上使用了多头注意力来**提取全局的时间依赖关系**。

在最后的分类器部分中使用了两层全连接层，并且以**交叉熵损失函数**作为损失函数进行训练。

### 数据集

| 数据集名字                    | 数据类型                        | 数据特点                                                     |
| ----------------------------- | ------------------------------- | ------------------------------------------------------------ |
| BCI Competition IV Dataset 2a | 运动想象（Motor Imagery）       | 四类运动想象（左手、右手、双脚、舌头），22 个 电极，250 Hz 采样率，9 名受试者，每类72个试次 |
| BCI Competition IV Dataset 2b | 运动想象（Motor Imagery）       | 两类任务（左手、右手想象），3个双极电极（C3、Cz、C4），250 Hz 采样率，9名受试者，每类120个试次 |
| SEED                          | 情绪识别（Emotion Recognition） | 三类情绪（积极、中性、消极），15名受试者，62个电极，采样率1000 Hz（降采样至200 Hz），每类15个试次 |

### 预处理

另外，会对每个通道的脑电数据进行**标准化**，实验方法是**被试依赖实验**。

| 数据集名字                    | 数据划分                                     |
| ----------------------------- | -------------------------------------------- |
| BCI Competition IV Dataset 2a | 一个session用于训练，另一个session用于测试   |
| BCI Competition IV Dataset 2b | 前三个session用于训练，后两个session用于测试 |
| SEED                          | 将每个样本按秒进行窗口切分，使用五折交叉验证 |

关于数据增强，传统的数据增强方法是往数据里添加高斯噪声或者进行裁剪，论文认为这种方法会进一步降低EEG的信噪比，所以这里使用的数据增强方法是对原始数据进行裁切后随机连接来生成新的数据。

### 结果分析

结果展示了EEG Conformer在三个数据集上都取得了SOTA结果，并且在几乎所有的被试结果中都是最好的。

![image-20250325193326425](https://gitee.com/fbanhua/figurebed/raw/master/images/20250325193326468.png)

![image-20250325193521282](https://gitee.com/fbanhua/figurebed/raw/master/images/20250325193521310.png)

### 消融实验

论文在数据集BCI Competition IV Dataset 2a上进行了消融实验。图中表明，在去除transformer部分或数据增强操作后模型分类精度会出现比较明显的下降。

![image-20250325193929674](https://gitee.com/fbanhua/figurebed/raw/master/images/20250325193929708.png)

### 参数灵敏度

论文在这一部分对模型的超参数设置进行了分析，包括：**注意力层数**、**注意力头数**和**池化层的kernel大小**。

![image-20250325194350422](https://gitee.com/fbanhua/figurebed/raw/master/images/20250325194350462.png)

模型的分类精度会随着注意力模块的添加出现明显的上升，但是后面对模块深度的添加就不太敏感，只有参数量直线上升。证明模型对注意力部分的深度参数不敏感，准确率波动较小，最高与最低准确率相差仅 **1.24%**。

![image-20250325195336389](https://gitee.com/fbanhua/figurebed/raw/master/images/20250325195336420.png)

与注意力层数类似，模型同样对注意力头数不敏感，在BCI Competition IV Dataset 2a上的准确率波动范围为 **1.43%**，在BCI Competition IV Dataset 2b上的准确率波动范围为 **1.02%**。

![image-20250325195545672](https://gitee.com/fbanhua/figurebed/raw/master/images/20250325195545713.png)

池化层核大小从 15 增加到 45 时，准确率显著提升 **13.08%**；池化层核大小从 45 增加到 135 时，准确率趋于平稳，没有显著提升。证明池化层核大小对模型的分类精度有显著影响。

### 模型可视化

论文首先使用了t-SNE对模型提取的特征进行了降维和可视化，可以看到，有 Transformer 模块的特征分布更加紧凑，类别之间的**边界清晰**，表明特征更具区分度。

![image-20250325200337036](https://gitee.com/fbanhua/figurebed/raw/master/images/20250325200337075.png)

关于注意力部分的结果，论文使用了一种叫Class Activation Mapping (CAM)的可视化技术生成注意力图，反映模型在不同**时间点**和**电极通道**上的关注度分布。将CAM权重与原始EEG地形图结合，形成 CAT 图。

论文认为，所有被试在任务初期（时间序列起始位置）激活较弱，可能对应运动意图的潜伏期。

![image-20250325200940778](https://gitee.com/fbanhua/figurebed/raw/master/images/20250325200940830.png)

论文还提到，这个可视化表明符合“对侧激活、同侧抑制”的神经机制（被试1和被试8）。

![image-20250325201822271](https://gitee.com/fbanhua/figurebed/raw/master/images/20250325201822332.png)

### 局限性

论文认为，当前实验主要聚焦于运动想象和情感识别等振荡型 EEG 数据，尚未验证模型在事件相关电位（ERP）等平稳模式 EEG 数据上的表现；模型也因为使用了transformer所以有较大的参数量；另外，实验是在每个被试上分开进行构建的，没有使用来自其他被试的信息。
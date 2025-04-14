**论文链接：**[Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a)

## 数据集

论文首先介绍了文章要用的数据集，分为两类。第一类是无限数据集Shapeset-3x2（即通过指定算法可以无限生成样本的数据集），每个样本由三角形、平行四边形、椭圆中的两个图形随机摆放构成，其中一个不会盖住另一个样本超过50%的面积。任务目标是**预测生成的图像中包含什么类别的图形**，难点在于需要找到不同图形在经过旋转、缩放等操作后仍然存在的**不变性**。

![Shapeset-3x2](https://gitee.com/fbanhua/figurebed/raw/master/images/20250413104434662.png)

第二类是有限数据集，这部分使用了MNIST、CIFAR-10和Small-ImageNet，即分别为手写数字识别、10个类别的图像识别和从高分辨率图像计算的灰度图像数据。

![Small-Imagenet](https://gitee.com/fbanhua/figurebed/raw/master/images/20250413105716662.png)

## 实验

### 实验设计

**模型结构**是全连接层，层数由1到5，每层1000个神经元，输出层使用softmax，**损失函数**使用负对数似然函数
$$
\mathrm{NLL}=-\log P(y|x),
$$
其中$x$表示输入的图像，$y$表示真实的标签（模型输出的即为真实标签的概率）。

**模型训练**使用SGD进行梯度下降，batch大小为10，即计算10个随机样本上的
$$
\frac{\partial-\log P(y|x)}{\partial\theta}
$$
后进行梯度更新$\theta\leftarrow\theta-\epsilon g$，其中学习率$\epsilon$是由验证集调整的超参数。每个模型都将根据验证集选取最好的**学习率与模型层数**。

**激活函数**选取了sigmoid、tanh和softsign进行对比。

模型的**参数初始化**服从均匀分布
$$
W_{ij}\sim U\Big[-\frac{1}{\sqrt{n}},\frac{1}{\sqrt{n}}\Big],
$$
其中$n$表示前一层的神经元个数，偏置参数则全部初始化为0（后文称这个初始化方法为旧初始化方法或原始初始化方法）.

### 激活函数结果对比

在**Sigmoid**作为激活函数的模型中，最后一层隐藏层的输出非常快的接近于0（饱和），这将导致模型更新缓慢；当最后一层隐藏层的输出开始去饱和时，别的层的输出结果会发生明显波动。论文认为，随机初始化的神经网络中前面的几层无法起到提取任务特征的作用，这将导致最后一层输出时更多依赖的是最后一层的偏置参数而不是由前几层计算提取特征所得到的隐藏层特征$h$。

简单来说，就是**在模型训练初期，对分类结果起作用更大的是输出层的偏置参数而不是由整体神经网络所提取的特征**。这种情况下模型将隐藏层的结果视作一种噪声，将进一步导致模型偏向于将隐藏层的输出推向0以减少噪声的影响从而形成恶性循环。另外，论文中特别提到使用RBM进行参数初始化的模型不会出现这种饱和情况，因为模型已经能提取一定的特征。

同时，把隐藏层的结果推向于0不一定是坏的，在使用tanh作为激活函数时这种结果是好的，因为它允许梯度传播到初始的层中。

![Sigmoid](https://gitee.com/fbanhua/figurebed/raw/master/images/20250413120338351.png)

在**Tanh**作为激活函数的模型中，神经网络中从第一层到最后一层依次发生饱和现象并开始往后面的层传播，但论文指出并未清楚为什么会发生该种情况。而以**softsign**作为激活函数的模型中，发生了与tanh激活函数类似的现象，但是倾向于先快后慢的变化，最后所有层一起向更大的权重移动。

![Tanh](https://gitee.com/fbanhua/figurebed/raw/master/images/20250413150658074.png)

另外，论文画出了使用tanh和softsign作为激活函数训练后的模型关于输入值所输出的结果结果分布。可以看出两种激活函数训练的模型隐藏层输出结果的分布是显然不同的。

![直方图](https://gitee.com/fbanhua/figurebed/raw/master/images/20250413151152265.png)

## 梯度分析

### 关于损失函数的梯度

在模型的梯度更新中，如果模型参数在某一点的梯度十分平缓的话，会出现梯度过小的情况从而导致参数更新速度缓慢。如果仅使用两个参数在随机输入上进行训练就得到了文中图5，可以观察到使用均方误差损失函数的平面显然要比使用交叉熵损失函数的平面平坦，这也说明在进行分类模型的训练时交叉熵损失函数可能会是收敛更快的选择。

![梯度绘制](https://gitee.com/fbanhua/figurebed/raw/master/images/20250413152022800.png)

### 梯度的理论分析

对一个人工神经网络来说，假设使用的是对称激活函数并且在0处的导数为1，使用$z^{i}$表示第$i$层神经网络的输入，$W^{i+1}$和$b^i$分别表示该层的权重参数和偏置参数，那么这一层计算的结果是
$$
\mathbf{s}^i=\mathbf{z}^iW^i+\mathbf{b}^i,
$$
在得到当前层输出也就是下一层神经网络的输入时，还要经过激活函数的计算，这里假设激活函数为$f$，那么这一层的最终输出表示为
$$
\mathbf{z}^{i+1}=f(\mathbf{s}^{i}).
$$
在进行反向传播需要计算关于某一层神经元的梯度时，假设是对第$i$层权重矩阵的参数进行求导，那么根据前面的公式，可以**推导出论文中的求导公式即式(3)**：
$$
\begin{align}
(输入激活函数前的结果)\quad \frac{\partial Cost}{\partial w_{l,k}^i}
&=\frac{\partial Cost}{\partial s^i}\frac{\partial s^i}{\partial w_{l,k}^i} \\
(实际计算结果只与第k个元素相关)\quad&=\frac{\partial Cost}{\partial s_k^i}\frac{\partial s_k^i}{\partial w_{l,k}^i}\\
(可直接由矩阵运算中取出系数)\quad&=\frac{\partial Cost}{\partial s_k^i}z_l^{i},
\end{align}
$$
类似的，其中关于$s_k^i$可以进一步计算得到**论文中的式(2)**：
$$
\begin{align}
\frac{\partial Cost}{\partial s_k^i}
&=\frac{\partial Cost}{\partial\mathbf{s}^{i+1}}\frac{\partial \mathbf{s}^{i+1}}{\partial\mathbf{z}^{i+1}}\frac{\partial \mathbf{z}^{i+1}}{\partial\mathbf{s}_{k}^{i}}\\

&=\frac{\partial Cost}{\partial\mathbf{s}^{i+1}}
\frac{\partial \mathbf{s}^{i+1}}{\partial\mathbf{z}_{k}^{i+1}}
\frac{\partial \mathbf{z}_{k}^{i+1}}{\partial\mathbf{s}_{k}^{i}}\\

&=\frac{\partial Cost}{\partial\mathbf{s}^{i+1}}W_{k,\bullet}^{i+1}f'(s_k^i),\\
\end{align}
$$
那么只要从输出层一路反向传播到输入层就可以获取到每一层的梯度结果来进行参数更新。

论文在这一步讨论中认为预激活值（激活函数的输入）处于接近0的范围，所以得到**论文中的式(4)**：
$$
f'(s_k^i)\approx1,
$$
**这一步假设非常重要，因为激活函数关于预激活值的参数约等于1表明激活函数在输入值附近可以近似理解为是恒等映射函数**，这将方便后续公式的推导：可以直接把激活函数值约等于预激活值，也就是在推导中**可以忽视激活函数的存在**。

另外，论文假设模型的权重参数初始化是以独立同分布进行初始化的，并且认为每个样本所输入的特征服从的分布的方差为均为$Var[x]$，那么现在计算第$i$层神经网络的中每个输入的均值，假设现在需要计算的是$s_{k}^i$的方差，也就是输入中第$k$个元素的方差，这里先计算出均值：
$$
\begin{align}
E[z_{k}^{i}]
&=E[s_{k}^{i}] \\
(偏置参数全部初始化为0)\quad&=E[\sum_{j=1}^{n_{i-1}}W_{kj}^{i-1}\cdot z_{j}^{i-1}]\\
(权重和输入特征均视为互相独立)\quad &=\sum_{j=1}^{n_{i-1}}E[W_{kj}^{i-1}]E[z_{j}^{i-1}] \\
(权重均值为0)\quad &=0,
\end{align}
$$
现在根据均值结果按照方差计算公式计算$s_{k}^i$的方差：
$$
\begin{align}
Var[z_{k}^{i}]
&=E[(z_{k}^{i}-0)^2] \\
&=E[(z_{k}^{i})^2] \\
&=E[(\sum_{j=1}^{n_{i-1}}W_{kj}^{i-1}z_{j}^{i-1})^2] \\
(交叉项期望为0)
&=\sum_{j=1}^{n_{i-1}}E[(W_{kj}^{i-1}z_{j}^{i-1})^2]  \\
(权重参数与输入特征互相独立)
&=\sum_{j=1}^{n_{i-1}}E[(W_{kj}^{i-1})^2]E[(z_{j}^{i-1})^2]  \\
(同一层的权重参数与输入特征服从对应的同一分布)
&=n_{i-1}\cdot Var[W^{i-1}]\cdot Var[z^{i-1}], \\
\end{align}
$$
这里假设权重参数中的每一个参数均服从同样的分布，以$Var[W^{i-1}]$符号表示其所服从分布的方差，$Var[z^{i-1}]$的含义在输入特征下相同（为什么每一层参数中的方差不一样？因为每一层参数的初始化所使用的分布依赖于神经元的个数）。通过这一步我们可以递推得到最终的当前层输入的方差计算公式，即**论文中的式(5)**：
$$
\operatorname{Var}\left[z^{i}\right]=\operatorname{Var}[x] \prod_{i^{\prime}=0}^{i-1} n_{i^{\prime}} \operatorname{Var}\left[W^{i^{\prime}}\right].
$$
现在我们计算损失函数关于$s^i$梯度的方差，根据前面推导的梯度公式
$$
\begin{align}
\frac{\partial Cost}{\partial s_k^i}
=\frac{\partial Cost}{\partial\mathbf{s}^{i+1}}W_{k,\bullet}^{i+1}f'(s_k^i)
\end{align}
$$
进行方差计算：
$$
\begin{align}
Var\Big[\frac{\partial Cost}{\partial s_{k}^i}\Big]
&=Var\Big[\frac{\partial Cost}{\partial\mathbf{s}^{i+1}}W_{k,\bullet}^{i+1}f'(s_k^i)\Big] \\
&=Var\Big[\frac{\partial Cost}{\partial\mathbf{s}^{i+1}}W_{k,\bullet}^{i+1}\Big] \\
&=Var\Big[\sum_{n_{i+1}}^{i+1}\frac{\partial Cost}{\partial\mathbf{s}_{j}^{i+1}}W_{kj}^{i+1} \Big] \\
&=\sum_{n_{i+1}}^{i+1}Var\Big[\frac{\partial Cost}{\partial\mathbf{s}_{j}^{i+1}}\Big]Var\Big[W_{kj}^{i+1} \Big] \\
&=n_{i+1} Var\Big[\frac{\partial Cost}{\partial\mathbf{s}^{i+1}}\Big]Var\Big[W^{i+1} \Big] ,
\end{align}
$$
这里**实际上蕴含了两点假设**：第一，假设了同一层中的关于输入或者计算说的结果的梯度也是全部服从同一个分布的；第二，梯度和权重参数服从的分布是互相独立的。另外，这里使用了和前面同样的表示方法来表示同一层中的同一类参数服从的是同一个分布，所以方差的表示忽略了下标。

使用推导得出的方差公式可以计算得到**论文中的式(6)**：
$$
Var\Big[\frac{\partial Cost}{\partial s^i}\Big]=Var\Big[\frac{\partial Cost}{\partial s^d}\Big]\prod_{i'=i}^dn_{i'+1}Var[W^{i'}],
$$
这里$d$表示当前推导公式的神经网络中一共有$d$层（但是实际代入一个简单模型计算时会发现这个$d$不太对劲，本文暂不讨论，仅对论文公式进行推导），把这个公式和关于模型中权重梯度的公式代入梯度计算中可以得到**论文中的式(7)**：
$$
\begin{aligned}
Var\Big[\frac{\partial Cost}{\partial w^{i}}\Big]&=\prod_{i^{\prime}=0}^{i-1}n_{i^{\prime}}Var[W^{i^{\prime}}]\prod_{i^{\prime}=i}^{d-1}n_{i^{\prime}+1}Var[W^{i^{\prime}}]\\&\times Var[x]Var\Big[\frac{\partial Cost}{\partial s^{d}}\Big],
\end{aligned}
$$
这一步中是简单的代入，所以不做推导了。

从前向传播的观点来看，我们希望保持信息的流动，也就是希望
$$
\forall(i,i'),\quad Var[z^i]=Var[z^{i'}],
$$
而从反向传播的观点来看，同样是希望
$$
\forall(i,i'),\quad Var\Big[\frac{\partial Cost}{\partial s^i}\Big]=Var\Big[\frac{\partial Cost}{\partial s^{i^{\prime}}}\Big].
$$
**为什么说保持方差一致就可以保持信息的流动？**在**前向传播**中，如果方差过大或方差过小都会导致激活函数的输出值接近饱和状态（比如无限接近1或者无限接近0），这种情况的激活函数值没有区分度；在**反向传播**中，方差过大会导致梯度爆炸，模型参数发生震荡无法收敛，方差过小则导致梯度消失，参数的更新几乎为0模型无法有效学习。

现在把前面推导得出的三条公式贴在这，后面需要用：
$$
\begin{align}
\operatorname{Var}\left[z^{i}\right]&=\operatorname{Var}[x] \prod_{i^{\prime}=0}^{i-1} n_{i^{\prime}} \operatorname{Var}\left[W^{i^{\prime}}\right], \\
Var\Big[\frac{\partial Cost}{\partial s^i}\Big]&=Var\Big[\frac{\partial Cost}{\partial s^d}\Big]\prod_{i'=i}^dn_{i'+1}Var[W^{i'}], \\
Var\Big[\frac{\partial Cost}{\partial w^{i}}\Big]&=\prod_{i^{\prime}=0}^{i-1}n_{i^{\prime}}Var[W^{i^{\prime}}]\prod_{i^{\prime}=i}^{d-1}n_{i^{\prime}+1}Var[W^{i^{\prime}}]\\&\times Var[x]Var\Big[\frac{\partial Cost}{\partial s^{d}}\Big],
\end{align}
$$
按照前面前向传播和反向传播中希望实现的方差效果，可以把问题转换为希望
$$
\begin{align}
&\forall i,\quad n_{i}Var[W^{i}]=1, \\
&\forall i,\quad n_{i+1}Var[W^{i}]=1,
\end{align}
$$
第一条保证了前向传播中**每一层计算结果的方差**恒等于输入特征的方差，第二条保证了反向传播中**关于每一层的输出的梯度的方差**均为最后一层梯度的方差，这两条共同保证了损失函数关于模型中**每一层参数的梯度的方差**均等于输入特征方差和最后一层梯度方差的乘积。

但是仔细观察，这相当于要求$Var[W^{i}]$既等于$1/n_{i}$又等于$1/n_{i+1}$，除非我们选取一个折中方案：
$$
\forall i,\quad Var[W^i]=\frac2{n_i+n_{i+1}},
$$
或者说，当**所有层之间的宽度均相同时**这个条件就自然可以实现了。特别的，如果所有层初始化所用的分布都相同时，假设每一层的梯度的方差都等于$\operatorname{Var}[W]$，那么关于预激活值的梯度则均表示为：
$$
\forall i,\quad Var\Big[\frac{\partial Cost}{\partial s^i}\Big]=\Big[nVar[W]\Big]^{d-i}Var\Big[\frac{\partial Cost}{\partial s^d}\Big],
$$
把其中最后一项替换为输入特征的方差即得到**论文中的式(13)**：
$$
\forall i,\quad Var\Big[\frac{\partial Cost}{\partial s^i}\Big]=\Big[nVar[W]\Big]^{d-i}Var[x],
$$
**这相当于假设损失函数关于模型最终输出层结果的梯度分布方差和输入特征分布的方差一致**。

同样的进行代入可以得到**论文中的式(14)**：
$$
\forall i,\quad Var\Big[\frac{\partial Cost}{\partial w^i}\Big]=\Big[nVar[W]\Big]^dVar[x]Var\Big[\frac{\partial Cost}{\partial s^d}\Big].
$$
笔者在阅读这里的时候有一个不解，就是在式(13)中$Var\Big[\frac{\partial Cost}{\partial s^d}\Big]$最后被替换成了$Var[x]$，但是在式(14)中又没有进行替换，这里是为什么呢？若您知道答案，欢迎分享您的见解～

但是这里也突出了一个问题，就是$n_{i}Var[W^{i}]=1$这个条件是十分严格的，如果不满足比如说小于1时，代入上面的公式计算就会发现随着网络深度的增加发生梯度消失。

在前文中提到，模型的**参数初始化**服从均匀分布
$$
W_{ij}\sim U\Big[-\frac{1}{\sqrt{n}},\frac{1}{\sqrt{n}}\Big],
$$
其中$n$表示每一层神经元个数都相同，根据这个分布可以算出
$$
nVar[W]=\frac13,
$$
显然，这个方差会在训练中导致梯度消失。基于这个初始化，**论文给出了一个建议分布**
$$
W\sim U\Big[-\frac{\sqrt{6}}{\sqrt{n_j+n_{j+1}}},\frac{\sqrt{6}}{\sqrt{n_j+n_{j+1}}}\Big],
$$
根据均匀分布的方差公式可以计算得到：
$$
\mathrm{Var}[W]=\frac{\left(\frac{2\sqrt{6}}{\sqrt{n_j+n_{j+1}}}\right)^2}{12}=\frac{\frac{24}{n_j+n_{j+1}}}{12}=\frac2{n_j+n_{j+1}},
$$
即前面提到的**折中方案**，这也是后面被称为**Xavier参数初始化**的方法。

### Xavier参数初始化方法实验分析

论文后面对新提出的初始化方法进行了实验，可以看出无论是前向传播中的激活值和反向传播中的梯度值都比原有方法的分布更分散，也就是达到了前面的目标：通过更有区分度的输出帮助模型学习到更有效的特征。

![新初始化方法结果](https://gitee.com/fbanhua/figurebed/raw/master/images/20250414100726775.png)

使用Xavier参数初始化训练的权重更加稳定，没有发生明显的变大或者变小；而使用原始初始化时权重发生了明显的变大。另外，论文发现即使原始初始化下模型参数发生了明显变化，但是梯度却没有发生明显变化。

![权重梯度及权重](https://gitee.com/fbanhua/figurebed/raw/master/images/20250414101155444.png)

论文对前面提到的数据集和激活函数基于原始初始化方法和Xavier初始化方法进行了实验。

![image-20250414101728215](https://gitee.com/fbanhua/figurebed/raw/master/images/20250414101728393.png)

同样的，论文绘制除了不同数据集下关于训练轮数（也就是模型见到的累计样本数）变阿虎的误差曲线。基本可以认为，使用了Xavier初始化方法的模型收敛速度和稳定性都比原始初始化方法要好，但还比不过使用去噪自动编码器进行无监督预训练的预训练权重微调的结果。

![误差曲线](https://gitee.com/fbanhua/figurebed/raw/master/images/20250414102610441.png)

## 总结

这篇论文对深度学习中参数初始化的影响进行了分析，并且根据方差上的分析提出了一种新的初始化方法。但是这篇论文中的一些假设比如每层输出结果中的元素全部独立同分布或者梯度和权重参数互相独立这种看着感觉很强的假设但是好像并没有从比较数学的角度去验证或者说明，可能后面还要去查一查这部分的论文深入了解一下。另外就是这里是对仅由全连接层构成的模型来进行分析，别的比如注意力机制等方法不知道是否需要另外的讨论。

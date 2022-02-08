### 一、欠/过拟合问题（Under fitting/Overfitting Problem）

![Fitting Problems-1](https://cdn.jsdelivr.net/gh/321hjd/ImageBed/MachineLearning/Logistic-Regression/Fitting-Problems-1.jpg)![Fitting Problems-2](https://cdn.jsdelivr.net/gh/321hjd/ImageBed/MachineLearning/Logistic-Regression/Fitting-Problems-2.jpg)

1. 欠拟合

   拟合偏差非常大，用于预测时误差也会非常大。

2. 过拟合

   方差非常大，即拟合曲线与训练数据拟合得非常好以至于曲线非常复杂，导致缺乏足够的数据来约束，不能很好地泛化到新的样本数据中。

3. 解决拟合问题

   * **减少特征的数量**
     * 人工选择
     * 自动选择算法
   * **正则化（Regularization）**
     * 保留所有的特征，但减少参数的数量级

### 二、正则化（Regularization）

1. 在优化目标函数中加入**惩罚项**以缩小参数值（$\lambda$为正则化参数）

   $\min_\limits{\theta}[\frac{1}{2m}\sum_\limits{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2+{\color{red}\lambda\sum_\limits{j=1}^n\theta_j^2}]$（一般不会用$\theta_0$，但影响不大）

   * **更小的参数值意味着更简单的假设函数和更平滑的拟合曲线。**

   * 但是正则化参数 $\lambda$ 不能太大，否则相当于只含$\theta_0$，会导致欠拟合

2. 例：
	> 如一个有3个参数的目标函数，在其后加入$\lambda(\theta_3+\theta_4)$项，且$\lambda$很大，则要使整个目标函数最小，必然要让$\theta_3,\theta_4$接近0，相当于忽略了这两个参数。
	>
	> $\min_\limits{\theta}[\frac{1}{2m}\sum_\limits{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2+{\color{red}\lambda(\theta_3^2+\theta_4^2)}]$

### 三、正则化线性回归

1. 梯度下降法

   * 目标函数

     $\min_\limits{\theta}\frac{1}{2m}[\sum_\limits{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2+{\lambda\sum_\limits{j=1}^n\theta_j^2}]$

   * 迭代过程

     Repeat{

     $\theta_0=\theta_0-\alpha\frac{1}{m}\sum_\limits{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_0$

     $\begin{align}\theta_j=&\theta_j-\alpha[\frac{1}{m}\sum_\limits{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j+{\color{red}\frac{\lambda}{m}\theta_j}]\\=&\theta_j(1-\alpha\frac{\lambda}{m})-\alpha\frac{1}{m}\sum_\limits{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j\end{align}$

     更新 $\theta_j,j=1,2,\dots,n$ 

     }

     其实就是每次迭代的时候都将 $\theta_j$ 缩小一点

2. 正规方程

   > 正则化还能解决样本数量小于特征数量时矩阵不可逆的问题。加入 $\lambda$ 矩阵项后，括号内的矩阵一定可逆

   ![Normal Func](https://cdn.jsdelivr.net/gh/321hjd/ImageBed/MachineLearning/Linear-Regression/Normal-Function.jpg)

### 四、正则化逻辑回归

1. 目标函数

   $\min_\limits{\theta}-\frac{1}{m}\left[\sum_\limits{i=1}^my^{(i)}\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))\right]+\frac{\lambda}{2m}\sum_\limits{j=1}^n\theta_j$

2. 迭代过程

   Repeat{

   $\theta_0=\theta_0-\alpha\frac{1}{m}\sum_\limits{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_0$

   $\begin{align}\theta_j=&\theta_j-\alpha[\frac{1}{m}\sum_\limits{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j+{\color{red}\frac{\lambda}{m}\theta_j}]\\=&\theta_j(1-\alpha\frac{\lambda}{m})-\alpha\frac{1}{m}\sum_\limits{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j\end{align}$

   更新$\theta_j,j=1,2,\dots,n$

   }

   其中$h_\theta(x)=\frac{1}{1+z^{-\theta^Tx}}$

### 五、高级优化算法的正则化应用

![Reguarization in Advanced Optimization](https://cdn.jsdelivr.net/gh/321hjd/ImageBed/MachineLearning/Logistic-Regression/Reguarization-in-Advanced-Optimization.jpg)


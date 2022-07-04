# Diffusion Model

从2020年开始扩散模型（Diffusion Models）出现并在图像生成领域取得了令人惊艳的效果。关于扩散模型的各种探索和应用也越来越多。在刚刚接触扩散模型时，被论文中长篇的数学论证困扰很久。因此在本文中写下自己对于这个模型的理解和感想，如有所失欢迎大家批评斧正。

1. 什么是扩散模型，如何理解扩散过程
2. 扩散模型和GAN生成模型的对比
3. 扩散模型的灵活应用

# 什么是扩散模型（Diffusion Models）

因为DDPM原文的推导过程很复杂，在本节中比较简单的介绍了这个过程帮助大家理解。如果希望了解更细节严谨的推导过程，请阅读原文。

扩散过程定义了一个双向的过程：**Forward diffusion process (q)**和**Backward diffusion process (p)**。下图展示了扩散过程的可视化效果[1]。

![Untitled](Diffusion%20Model%203a636ca9a6504cf2bd8f3fd9819a64bc/Untitled.png)

扩散模型[1]提出假设有一张初始的图片$X^0$，在每一步$t$都在其中叠加少量的独立同分布的噪声$\mathbf{z}\sim\mathcal{N}(0,\mathbf{I})$。采样$T$步后，我们可以得到一组有噪声的图片$\{X^1,X^2,...,X^T\}$。当$t\rightarrow\infty$，$X^T\sim\mathcal{N}(0,\mathbf{I})$。
对于任意的$X^0$都可以通过这个Forward diffusion process得到符合独立同分布的$X^T$。

$$
\begin{align}
X^{t+1}=X^{t}+\mathbf{z}_t\\
X^T=\sum_{i=1}^{T}{X^i+\mathbf{z}_i}
\end{align}
$$

理论上，利用这个Forward diffusion process可以生成一组带有噪声的图像。利用这组数据训练生成器，可以实现从一个有噪声的图像估计出其中叠加的噪声，并恢复出真实的图像。那么这个生成器就可以从$X^T\sim\mathcal{N}(0,\mathbf{I})$恢复出一张清晰的图像。此时，生成器$G$的所有参数$\theta$应该通过：

$$
\begin{align}
\mathcal{L}&=X^t-\hat{X^t}\\
X^t&=\sum_{i=1}^{t}\{{X^i+\mathbf{z}_i}\}\\
\hat{X^t}&=X^T-p_\theta(X^T)p_\theta(X^T\vert X^{T-1})p_\theta(X^{T-2}\vert X^T,X^{T-1})...p_\theta(X^{t+1}\vert X^T,X^{T-1}...X^{t+2})
\end{align}
$$
优化。即，计算真实噪声的图像$X^t$和生成的图像$\hat{X^t}$之间的差距。

但是直接通过这种方式训练是非常困难的。因为：

1. 训练集过于庞大而无法获得。想要获得完全真实的训练集，我们需要对每张图像都添加$T$步独立同分布的噪声得到真实的训练集$\{X^T,X^{T-1},\cdots,X^0\}$。
2. prediction过程过于复杂，网络$G$难以训练。即使获得了真实的训练集，在训练图像时，因为每个图像在每一步$t=\{T,T-1,\cdots,t+1\}$添加的都是独立同分布的噪声，那$G$需要对每个图像都从$X^T$开始估计在每一步$t$叠加的噪声，并且。这样的开销太大且难以优化模型参数。

为了简化训练的难度Jonathan Ho[1]提出利用一组方差$\beta=\{ \beta_t\in(0,1) \}_{t=1}^T$来控制这个过程，即：

$$
\begin{align}
q(X^t\vert X^{t-1})=\mathcal{N}(X^t;\sqrt{1-\beta_t}X^{t-1},\beta_t\mathbf{I})\\
q(X^{1:T}\vert X^0)=\prod_{t=1}^Tq(X^t\vert X^{t-1})
\end{align}
$$

利用reparameterization trick，另$\alpha_t=1-\beta_t$，$\bar{\alpha}_t=\prod_{i=1}^T{\alpha_i}$，则Forward diffusion process可以表示为：

$$
\begin{align}
q(X^t\vert X^0)=\mathcal{N}(X^t;\sqrt{\bar{\alpha}_t}X^0,(1-\bar{\alpha}_t)\mathbf{I})
\end{align}
$$

通过一组固定的方差变量来控制Forward diffusion process，我们可以对于每个图像只采样一个噪声，在不同时间步骤上通过$\beta$为这个噪声增加幅度，以此来实现近似拟合forward diffusion process。

此时Backward diffusion process可以表示为：

$$
\begin{align}
p(X^t\vert X^{t-1}) &= \mathcal{N}(X^t;\mu_\theta(X^T),\textstyle{\sum}_\theta(X^t,t))\\
p_\theta(X^{0:T})&=p(X^T)\prod_{t=1}^Tp_\theta(X^{t-1}\vert X^t)
\end{align}
$$

生成器$G$的优化目标就变为了在当前状态下（$X^t,t$）估计采样的噪声的分布。训练目标可以表示为：

$$
\begin{align}
\mathcal{L}=\epsilon - \mathbf{z}_\theta(\sqrt{\bar{\alpha}_t}X^0+\sqrt{1-\bar\alpha_t}\epsilon;t)
\end{align}
$$

由此扩散模型可以通过一个简单的MSE loss实现优化。

## 和Score-based Models的联系

在扩散模型相关的论文中，会发现有一些论文会提到Score matching with Langevin dynamics (SMLD)的概念。
一开始我也有困惑关于他们之间的联系和如何理解这两个生成模型。
因此在这里简单介绍一下。

在论文[6]中，Song从理论基础分析并对比了这两种生成模型。
并且在文中提出了利用随机微分方程（SDEs）来统一表示为基于分数的扩散模型（score-based generative model）。扩散过程可以表示为：

$$
\mathrm{d}{X^t}=\mathbf{f}(X^t,t)\mathrm{d}t+g(t)\mathrm{d}\mathbf{w}
$$

根据我的个人的理解，无论扩散过程如何变形，他们insight都是相似的，即利用生成器$G$根据当前的状态$X^t$和时间步骤$t$估计相对于$X^{t-1}$的噪声$\epsilon$。

# 和其他生成模型的对比

目前主流的生成模型有VAE，Flow-based，还有最流行的GAN。这里讨论的主要对比对象是GAN模型。

GAN在很多领域都取得了令人惊艳的效果，比如图像生成，图像编辑。目前这些在各个任务中应用广泛的GAN模型并不是最原始的GAN模型，而是一种更广泛意义的生成模型，即generative model+discriminative training。

discriminative training带来了一些问题：因为需要同时训练一个判别器$D$，因为训练成本过高，容易造成mode coolapse；会surpress detail生成结果。与此同时discriminative training也带来了很多好处。通过这种训练方式生成的图像结构信息表现的更好，这个优点在人脸上最为明显；实际部署成本很低，因为训练好的生成器可以从latent space中很短的时间内推断模型。尤其是对比起扩散模型在inference时通过$T$步循环才能生成图像，效率上会高很多。

从扩散过程也可以看出，扩散模型的生成器在这里连接$p_{data}$和先验分布$p_{prior}\sim\mathcal{N}(0,\mathbf{I})$，理论上可以更好的拟合真实数据分布。
并且扩散模型的每一步都是随机采样，所以在处理非结构化信息上表现得会更好。
除了这些优点，在论文[2]中提出了一个概念即扩散过程是可引导的（guidance diffusion），这个特性使得扩散模型可以很容易的拓展到更多的应用。

![DdffusionvsGAN](Diffusion%20Model%203a636ca9a6504cf2bd8f3fd9819a64bc/Untitled%204.jpg)

DiffusionCLIP中与GAN模型的对比图。对于结构化信息不明显的图像基于Diffusion的方法处理效果更好


# 灵活的应用（Conditional Generation）
生成的图像的真实度和生成过程可控性是评价生成任务中非常重要的两个指标。
论文[2]提出因为扩散模型的每一步都是随机采样，所以如果为扩散过程的添加梯度，就可以变成可引导的扩散过程（guidance diffusion）。
他们利用了一个任务来证明这一点。
在不重新训练生成器的情况下，生成一个具有特定特征的图像。

论文[2]提出只需要先训练一个分类器区分特定特征，然后利用分类器的梯度来调节引导扩散采样过程朝向任意类标签。这样可以实现在不影响生成器的效果同时，实现生成带有特定特征图像的目的。此外训练分类器的成本比重新训练生成器要更小。
这一灵活的特性使得扩散模型也可以更容易地与其他任务结合，比如语义信息，图像编辑，风格迁移。



![基于文本的扩散模型生成图像[4]](Diffusion%20Model%203a636ca9a6504cf2bd8f3fd9819a64bc/Untitled%201.png)
<center>基于文本的扩散模型生成图像[4]<center>


![基于stroke的扩散模型生成图像[3]](Diffusion%20Model%203a636ca9a6504cf2bd8f3fd9819a64bc/Untitled%202.png)
<center>基于stroke的扩散模型生成图像[3]<center>

![基于CLIP的扩散模型生成图像](Diffusion%20Model%203a636ca9a6504cf2bd8f3fd9819a64bc/Untitled%203.png)
<center>基于CLIP的扩散模型生成图像[5]<center>



### Reference

1. DDPM

2. Guidance DDPM

3. SDEdit

4. Latent Diffusion

5. DiffusionCLIP

6. sohi-dickstein

7. SDE-Yang Song

8. Score-base Yang SOng
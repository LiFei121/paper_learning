[TOC]

# 论文

## 2025

### [LLM-DR: A Novel LLM-Aided Diffusion Model for Rule Generation on Temporal Knowledge Graphs | Proceedings of the AAAI Conference on Artificial Intelligence](https://ojs.aaai.org/index.php/AAAI/article/view/33249)

conference: AAAI2025

idea:

code: :heavy_multiplication_x:

### [**DPCL-Diff: Temporal Knowledge Graph Reasoning Based on Graph Node**Diffusion Model with Dual-Domain Periodic Contrastive Learning](https://arxiv.org/pdf/2411.01477)

conference: AAAI2025

idea:主要得创新应该是引入了扩散模型，但未公开代码，本身又没接触过扩散模型，估计难以复现。

code::heavy_multiplication_x:

### [Tackling Sparse Facts for Temporal Knowledge Graph Completion](file:///F:/git_learning/paper_learning/paper/TacklingSparseFactsforTemporalKnowledgeGraphCompletion.pdf)

conference: WWW2025

idea: 这篇论文的核心思想是：为解决时序知识图谱（TKG）中因事实稀疏导致的预测不准问题，提出一种即插即用的自适应邻域增强层（ANEL），通过动态挖掘同时间窗的潜在相似邻居（即使无显式连接），并基于实体邻居数量自适应加权融合潜在信息（稀疏实体依赖更强，稠密实体依赖更弱），显著提升稀疏实体的表示质量，从而提升补全性能。

核心公式

最终实体嵌入由基模型嵌入与潜在信息加权融合： $$ e_{s,t} = f(s) + \phi(s) \cdot g(s) $$ 

- $f(s)$：基模型生成的实体嵌入（如 REGCN 的输出） 
- $g(s)$：潜在邻居聚合信息（通过 GAT 计算）
- $\phi(s)$：自适应权重，动态调节潜在信息贡献：  $$  \phi(s) = \frac{1}{1 + \exp(|N_s|)}  $$  ，这个公式可以借鉴一下
- $|N_s|$ 是实体 $s$ 在时间 $t$ 的显式邻居数  
- 稀疏实体（$|N_s|$ 小）：$\phi(s) \to 1$，高度依赖潜在信息
- 稠密实体（$|N_s|$ 大）：$\phi(s) \to 0$，弱化潜在信息，避免噪声

code: :heavy_multiplication_x:，代码失效了

## 2024

### [Large Language Models-guided Dynamic Adaptation for Temporal Knowledge Graph Reasoning](https://proceedings.neurips.cc/paper_files/paper/2024/file/0fd17409385ab9304e5019c6a6eb327a-Paper-Conference.pdf)

conference: NIPS2024

idea:这篇论文受说实话有点整合论文Temporal Knowledge Graph Forecasting Without Knowledge Using In-Context Learning - ACL Anthology和TLogic: Temporal Logical Rules for Explainable Link Forecasting on Temporal Knowledge Graphs | Proceedings of the AAAI Conference on Artificial Intelligence这两篇论文。论文和核心思想是首先采用TLogic中提取规则的方法，然后再评估每个关系的相似度，将这两个东西放到LLM中去，让LLM更新TLogic提取到的规则。后面再就是，这篇论文用到了验证集，验证集的主要功能其实是评估在训练数据中提取到的规则是否适用于验证集，模型会再一次提取验证集中的规则来更新在训练集中提取到的规则，模型的最后一部分就是根据规则进行推理，此外最后一部分其实是由两个模块组成的，一个是基于规则的推理，一个是基于图神经网络的推理。所以说作者提出的方法其实是方法大杂糅。

code:  ✔

### [ONSEP: A Novel Online Neural-Symbolic Framework for Event Prediction Based on Large Language Model](https://aclanthology.org/2024.findings-acl.378/)

conference: ACL2024

idea: 这篇论文其实思想还是十分简单的，先初步提取规则，然后让大模型判断规则是否合理，也会使用当前的数据不断地更新规则。这篇论文提前规则和TLogic论文提取规则不同，对于一个四元组，他是直接在历史数据中筛选头实体和尾实体一样的四元组，然后认为这个四元组对应的关系实体和原来的关系实体有关系。如何也会计算一些关系的置信度，总而言之计算方法和TLogic是有点不一样，而且他提前的规则其实就是只有1步的，感觉很粗糙。

code: :x:，论文中写了开源，但是GitHub仓库已经是空的了

### [LLM4DyG: Can Large Language Models Solve Spatial-Temporal Problems on Dynamic Graphs?](https://dl.acm.org/doi/pdf/10.1145/3637528.3671709)

conference: KDD2024

idea: 

code: ✔

### [AUnified Temporal Knowledge Graph Reasoning Model Towards Interpolation and Extrapolation](https://aclanthology.org/2024.acl-long.8.pdf)

conference: ACL2024

idea: 

## 2023

### [Temporal Knowledge Graph Forecasting Without Knowledge Using In-Context Learning - ACL Anthology](https://aclanthology.org/2023.emnlp-main.36/)

conference: EMNLP2023

idea:

code: ✔



## 2022

### [Graph Hawkes Transformer for Extrapolated Reasoning on Temporal Knowledge Graphs](https://aclanthology.org/2022.emnlp-main.507.pdf)

conference: EMNLP2022

idea:

code: ✔

### [Search to Pass Messages for Temporal Knowledge Graph Completion](https://aclanthology.org/2022.findings-emnlp.458.pdf)

conference: EMNLP2022

idea:

code: ✔

### [**Towards Event Prediction in Temporal Graphs**](https://www.vldb.org/pvldb/vol15/p1861-tian.pdf)

conference: VLDB2022

idea:

code: :heavy_multiplication_x:

### [TLogic: Temporal Logical Rules for Explainable Link Forecasting on Temporal Knowledge Graphs | Proceedings of the AAAI Conference on Artificial Intelligence](https://ojs.aaai.org/index.php/AAAI/article/view/20330)

conference: AAAI2022

idea:应该是第一篇在时序知识图谱推断任务中使用规则进行推理的论文，后面的很多基于规则的论文都是以此为基础的。论文所提方法由**Rule Learning**、**Rule Application**和**Candidate Ranking**三个模块组成。第一个模块是根据训练数据进行提取规则，提取规则核心就是用到了一个random walk的概念，具体而言就是从每一个关系出发，随机选择一个是这个关系的四元组，然后从这个四元组出发，沿着时间轴反方向寻找能够回到这个四元组的路径。如果能够回到，则表明这个规则是成立的。随机游走完全是跟着时间来的，时间距离上一个四元组越近，则被选择的概率越大。第二模块就是应用第一步提取到的规则，具体而言，对于其中一个的测试四元组，选择对应这个规则关系对应的body_rules，然后找到满足关系的路径，最后一个实体即被认为是候选实体，然后计算每个候选实体的score。第三个模块就是评估方法的性能，直接计算hit@k和MRR，hit@k越大越好，MRR越小越好。

<img src="image\屏幕截图 2025-06-26 161430.png" alt="屏幕截图 2025-06-26 161430" style="zoom:75%;" />

code: ✔,代码复现结果和论文中基本一致。








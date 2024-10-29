# QLoRA **4-bit NormalFloat Quantization(NF4)**

## 背景
<img src="https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/LLM3.png" style="zoom:30%;">

语言模型一直在变大，由于这些模型很大，因此它们很难在一般的设备上运行。举个例子，仅推理 BLOOM-176B 模型，你就需要 8 个 80GB A100 GPU (每个约 15,000 美元)。而如果要微调 BLOOM-176B 的话，你需要 72 个这样的 GPU！更大的模型，如 PaLM，还需要更多资源。

## 前置知识
### 数据类型
在介绍NF4量化之前，我们先需要理解一下不同的数据类型，这些数据类型在机器学习中被称为精度。模型的大小通常由其参数量及其精度决定，精度通常为 float32、float16 或 bfloat16之一，如下图所示。

<img src="https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/tf32-Mantissa-chart-hi-res-FINAL.png" style="zoom:70%;">

### 量化
在理解NF4量化之前，我们还需要理解量化。对于量化，简而言之就是以较少的位数来存储信息，并使模型仍达到一个不错的效果。量化分为量化和反量化两个阶段。下图展示的是将FP16类型的数据量化为INT4类型的数据，并反量化INT4类型的数据来估计原本对应的数据。  
**注意：在反量化之后得出的估计数据与原数据有误差！**

<img src="https://github.com/zaizaiddg/CIML/blob/cc874714181745625442cb4490d44349a59daec5/%E5%9B%BE%E7%89%87/%E9%87%8F%E5%8C%96.png?raw=true" style="zoom:70%;">  



**注意：得到这种分位数结果的前提是假设[-1,1]是均匀的分布，下图为[-1,1]之间均匀分布的概率密度函数。**  

<img src="https://github.com/zaizaiddg/CIML/blob/master/%E5%9B%BE%E7%89%87/simplequant-int4.png?raw=true" style="zoom:70%;">

## 4-bit NormalFloat Quantization(NF4)

### 基本概念  
4-bit NormalFloat (NF4) 是一种用于量化神经网络权重的数据类型。它的设计旨在有效地表示以零为中心的正态分布数据，从而在保持较低内存需求的同时尽量减少信息损失。
>预训练的神经网络权重具有**以零中心的正态分布的性质**（因此NF4数据类型比INT4和Float4更适用于微调神经网络），通过缩放标准差σ，使其权重落在[-1,1]上，分位数也落在这个范围内。缩放权重后，0附近会出现大部分数据，而NF4对0有精确的表示（INT4、FLOAT4则没有），因此即使其只有4bit但是还是尽量减少了量化后的信息损失。

### 如何构建NF4数据类型
下图展示了构建NF4数据类型的步骤，具体分为5步。NF4最多可存放16个分位数,因此[-1,0)种包含7个负分位数[0,1]包含8个分位数（其中有一个是0，其他为正分位数）  
<img src="https://github.com/zaizaiddg/CIML/blob/master/%E5%9B%BE%E7%89%87/nf4.png?raw=true" style="zoom:50%;">

>step1：从0.56-0.97等距离的选择8个值  
>step2：从0.57-0.97等距离的选择7个值  
>step3：计算step1中8个值的z-score值，计算step2中7个值的z-score值并取负  
>step4：连接这8个z-score值和7个z-score值（取负）连接点的值取0  
>step5：对z-score值进行标准化使其落入[-1,1]这个区间，这16个点即为NF4在[-1,1]中的分位点
```python
# NF4数据类型在[-1,1]之间对应的16个分位点的确切数值
quantiles =  [
-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
-0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
```

构造的NF4数据类型的概率密度函数应该如下图所示。
<img src="https://github.com/zaizaiddg/CIML/blob/master/%E5%9B%BE%E7%89%87/nf4_2.png?raw=true" style="zoom:70%;">

## 总结
NF4 的设计和实现是为了更有效地量化以零为中心的正态分布数据。通过使用标准正态分布的分位数函数，NF4 能够在保持较低位数的同时，尽量减少信息损失，从而提高模型的性能和效率。

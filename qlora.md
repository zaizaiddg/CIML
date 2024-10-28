# QLoRA

## 4-bit NormalFloat Quantization

### Backgound
<img src="https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/LLM3.png">

语言模型一直在变大，由于这些模型很大，因此它们很难在一般的设备上运行。举个例子，仅推理 BLOOM-176B 模型，你就需要 8 个 80GB A100 GPU (每个约 15,000 美元)。而如果要微调 BLOOM-176B 的话，你需要 72 个这样的 GPU！更大的模型，如 PaLM，还需要更多资源。

### Introduction
为了解决这个问题，作者提出了QLoRA这种方法。其中4-bit NormalFloat Quantization（NF4）量化为QLoRA中的核心创新。

在介绍NF4之前我们先介绍一下量化，如下图所示。
<img src="">






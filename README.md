<div align="center">
<img src="fig/logo.png" alt="logo" width="2200" style="display:block; margin-bottom:0;"/>
<img src="https://img.shields.io/badge/version-1.0.1-6395ED" alt="version"/>
<img src="https://img.shields.io/badge/license-MIT-9ACD32" alt="license"/>
<a href="https://arxiv.org/abs/2501.08001"><img src="https://img.shields.io/badge/Preprint'25-EE4C2C" alt="preprint"/></a>
<a href="https://2026.emnlp.org/"><img src="https://img.shields.io/badge/EMNLP-2026%20Main-B57EDC" alt="EMNLP 2026 Main"/></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch"/></a>
<img src="https://img.shields.io/github/stars/sunshy-1/JuDi?style=social" alt="stars"/>
</div>

<br/>



This is the Pytorch implementation for our *EMNLP'26 Main Conference* paper: [**JuDi: Revisiting Judge Decoding from First Principles via Training-Free Distributional Divergence**](https://arxiv.org/abs/2601.04766). 

## Abstract
<div style="text-align: justify;">
  Judge Decoding accelerates LLM inference by relaxing the strict verification of Speculative Decoding, yet it typically relies on expensive and noisy supervision. In this work, we revisit this paradigm from first principles, revealing that the ``criticality'' scores learned via costly supervision are intrinsically encoded in the draft-target distributional divergence. We theoretically prove a structural correspondence between learned linear judges and Kullback-Leibler (KL) divergence, demonstrating they rely on the same underlying logit primitives. Guided by this, we propose a simple, training-free verification mechanism based on KL divergence. Extensive experiments across reasoning and coding benchmarks show that our method matches or outperforms complex trained judges (e.g., AutoJudge), offering superior robustness to domain shifts and eliminating the supervision bottleneck entirely.
<div> 
<br>

![Framework](fig/framework.png)

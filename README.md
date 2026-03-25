# FairBranch

[![MAI_BIAS toolkit](https://img.shields.io/badge/MAI_BIAS-⚖️_AI_fairness_tool-white)](https://mammoth-eu.github.io/mammoth-commons/index.html)

This software is part of MAI-BIAS; a low-code toolkit for
fairness analysis and mitigation, with an accompanying suite of coding
tools. Our ecosystem operates in multidimensional and multi-attribute
settings (safeguarding multiple races, genders, etc), and across multiple
data modalities (like tabular data, images, text, graphs). Learn more
[here](https://mammoth-eu.github.io/mammoth-commons/index.html).

## 👥 Who is this for?

- **ML engineers and data scientists** training neural networks on multi-task problems who need built-in fairness constraints.
- **AI researchers** studying bias mitigation in multi-task and multi-attribute settings across vision and tabular data.
- **Compliance and ethics teams** seeking evidence that bias was addressed at the training stage, not patched after deployment.
- **Policymakers and auditors** evaluating AI systems: ask whether the organisation whose system you are reviewing used a bias mitigation method during training. If they use FairBranch, they can show you quantitative fairness scores across demographic groups as evidence of due diligence with no technical background required to interpret the results.

## ✨ About

FairBranch is a fairness-aware multi-task learning framework for neural
networks. AI systems trained on real-world data frequently inherit societal
biases, producing decisions that disadvantage people based on protected
characteristics such as gender or race. FairBranch addresses this at the
architectural level: instead of patching a biased model after the fact, it
builds fairness directly into how the network learns.

The core idea is a **branching mechanism**: when the model detects that
learning one task conflicts with fair treatment of a protected group, it
routes that task through a dedicated sub-network (branch), reducing harmful
interference. A fairness regularisation term further penalises disparate
performance across groups throughout training.

FairBranch has been validated on both **image classification** and **tabular
data** settings and benchmarked against leading multi-task fairness methods.
Results show that fairer models can be achieved without sacrificing predictive
accuracy. The method is published at IJCNN (IEEE International Joint
Conference on Neural Networks); the paper and conference presentation are
available in this repository.

Any organisation deploying AI across heterogeneous user populations whether in
hiring, lending, healthcare triage, or image-based services can use
FairBranch to demonstrate that fairness was an explicit training objective,
not an afterthought.

## 🚀 Highlights

⚖️ Fairness built into training: not applied as a post-hoc fix  
🌿 Branching architecture reduces harmful gradient interference between tasks  
👥 Multi-attribute support: handles multiple protected characteristics simultaneously  
🖼️ Multi-modal: validated on both tabular data and image classification  
📄 Peer-reviewed: published at IJCNN 2024  

## 🔗 Material

- [IJCNN Paper](./IJCNN_FairBranch.pdf) - full method description and experimental results
- [Conference Presentation](./WCCI-IJCNN_FairBranch_Presentation.pdf) the slides from the WCCI-IJCNN presentation
- [`FairBranch/`](./FairBranch/) the notebooks with step-by-step guidance for vision and tabular setups

## ⚡ Quick start

Detailed step-by-step notebooks are provided in the [`FairBranch/`](./FairBranch/)
directory, covering:

- **Vision setup** - multi-task image classification with fairness constraints
- **Tabular setup** - structured data with multiple protected attributes
- **STL baseline training** - single-task baselines for benchmarking

## Competitors & Baselines

The following methods are used as baselines in our benchmarking:

1. **FAFS**: https://github.com/luyongxi/deep_share
2. **TAG**: https://github.com/google-research/google-research/tree/master/tag
3. **PCGRad**: https://github.com/WeiChengTseng/Pytorch-PCGrad
4. **Recon**: https://github.com/moukamisama/Recon
5. **L2T-FMT**: https://github.com/arjunroyihrpa/L2TFMT
6. **WB-fair**: https://github.com/phi-ra/FairMultitask/tree/main

## 📜 Attributions

```
@inproceedings{roy2024fairbranch,
  title={FairBranch: Mitigating Bias in Multi-Task Learning via Branch Selection},
  author={Roy, Arjun and others},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2024}
}
```

**Project:** [MAMMOth](https://mammoth-ai.eu/)  
**Maintainer:** Arjun Roy


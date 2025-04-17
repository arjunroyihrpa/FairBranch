<link rel="stylesheet" href="/FairBranch/assets/css/style.css">

## Abstract
The generalisation capacity of  Multi-Task Learning (MTL) suffers when unrelated tasks negatively impact each other by updating shared parameters with conflicting gradients. This is known as negative transfer and leads to a drop in MTL accuracy compared to single-task learning (STL). Lately, there has been a growing focus on the fairness of MTL models, requiring the optimization of both accuracy and fairness for individual tasks. Analogously to negative transfer for accuracy, task-specific fairness considerations might adversely affect the fairness of other tasks when there is a conflict of fairness loss gradients between the jointly learned tasks - we refer to this as bias transfer. To address both negative- and bias-transfer in MTL, we propose a novel method called FairBranch, which branches the MTL model by assessing the similarity of learned parameters, thereby grouping related tasks to alleviate negative transfer. Moreover, it incorporates fairness loss gradient conflict correction between adjoining task-group branches to address bias transfer within these task groups. Our experiments on tabular and visual MTL problems show that FairBranch outperforms state-of-the-art  MTLs on both fairness and accuracy. 
![image](https://github.com/user-attachments/assets/fbb95019-cd54-43bc-b380-9e1fc5ca55d5)

## Important Links
<div style="margin: 20px 0;">
  <a href="https://ieeexplore.ieee.org/abstract/document/10651221" class="button">Paper</a>
  <a href="https://github.com/arjunroyihrpa/FairBranch" class="button">Code</a>
  <a href="./WCCI-IJCNN_FairBranch_Presentation.pdf" class="button">Slides</a>
  <a href="./WCCI-IJCNN_FairBranch_Arjun_Roy.pptx" class="button">PPT</a> 
  <a href="https://youtu.be/UK1ke5_AV_g?si=kXk_mAof2cw4W1Od" class="button">Video</a> 
</div>
---
FairBranch is maintained by [arjunroyihrpa](https://github.com/arjunroyihrpa)
📧 [arjunroyihrpa@gmail.com](mailto:arjunroyihrpa@gmail.com)
🪩 [www.arjunroy.info](https://www.arjunroy.info)

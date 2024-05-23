# FairBranch: Mitigating Bias Transfer in Fair Multi-task Learning
## Abstract
The generalisation capacity of  Multi-Task Learning (MTL) suffers when unrelated tasks negatively impact each other by updating shared parameters with conflicting gradients. This is known as negative transfer and leads to a drop in MTL accuracy compared to single-task learning (STL). Lately, there has been a growing focus on the fairness of MTL models, requiring the optimization of both accuracy and fairness for individual tasks. Analogously to negative transfer for accuracy, task-specific fairness considerations might adversely affect the fairness of other tasks when there is a conflict of fairness loss gradients between the jointly learned tasks - we refer to this as bias transfer. To address both negative- and bias-transfer in MTL, we propose a novel method called FairBranch, which branches the MTL model by assessing the similarity of learned parameters, thereby grouping related tasks to alleviate negative transfer. Moreover, it incorporates fairness loss gradient conflict correction between adjoining task-group branches to address bias transfer within these task groups. Our experiments on tabular and visual MTL problems show that FairBranch outperforms state-of-the-art  MTLs on both fairness and accuracy. 

## Important Links
<div style="margin: 10px 0;">
  <a href="./IJCNN_FairBranch.pdf" class="button">Paper</a>
  <a href="https://github.com/arjunroyihrpa/FairBranch" class="button">Code Repository</a>
</div>


## Abstract
The generalisation capacity of  Multi-Task Learning (MTL) suffers when unrelated tasks negatively impact each other by updating shared parameters with conflicting gradients. This is known as negative transfer and leads to a drop in MTL accuracy compared to single-task learning (STL). Lately, there has been a growing focus on the fairness of MTL models, requiring the optimization of both accuracy and fairness for individual tasks. Analogously to negative transfer for accuracy, task-specific fairness considerations might adversely affect the fairness of other tasks when there is a conflict of fairness loss gradients between the jointly learned tasks - we refer to this as bias transfer. To address both negative- and bias-transfer in MTL, we propose a novel method called FairBranch, which branches the MTL model by assessing the similarity of learned parameters, thereby grouping related tasks to alleviate negative transfer. Moreover, it incorporates fairness loss gradient conflict correction between adjoining task-group branches to address bias transfer within these task groups. Our experiments on tabular and visual MTL problems show that FairBranch outperforms state-of-the-art  MTLs on both fairness and accuracy. 

## Important Links
<div style="margin: 10px 0;">
  <a href="./IJCNN_FairBranch.pdf" class="button">Paper</a>
  <a href="https://github.com/arjunroyihrpa/FairBranch" class="button">Code Repository</a>
</div>
<style>
  body, .container {
  max-width: 1500px; /* Adjust the max-width as needed */
  margin: 0 auto; /* Center the body with auto margins */
  padding: 20px; /* Optional padding for better appearance */
  width: 100% !important; /* Ensure full width */
}
.abstract-button-style {
  font-size: 1.2em; /* Adjust font size */
  line-height: 1.6; /* Adjust line height for better readability */
  text-align: justify; /* Justify the text */
  margin: 10px 0; /* Add some margin above and below */
  padding: 10px; /* Add padding for better spacing */
  text-align: center;
  border-left: 1px solid #007bff; /* Optional: Add a left border for styling */
  border-radius: 5px; /* Similar to the button class */
  box-shadow: 0 4px #999; /* Similar to the button class */
}
.button {
  display: inline-block;
  padding: 10px 20px;
  font-size: 16px;
  cursor: pointer;
  text-align: center;
  text-decoration: none;
  outline: none;
  color: #fff;
  background-color: #007bff;
  border: none;
  border-radius: 5px;
  box-shadow: 0 4px #999;
}

.button:hover {background-color: #0056b3}

.button:active {
  background-color: #0056b3;
  box-shadow: 0 2px #666;
  transform: translateY(2px);
}
figure {
  margin: 1em 0;
}

figcaption {
  text-align: center;
  font-style: italic;
  color: #555;
}
</style>

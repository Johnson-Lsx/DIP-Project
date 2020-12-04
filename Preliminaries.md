#### 预备知识

1. 分类算法的常见评价指标

   + 假设我们分类的目标只有两类，计为正例(positive)和负例(negative)：

     + True positives(TP): 被正确地划分为正例的个数，即实际为正例且被分类器划分为正例的实例数（样本数）

     + False positives(FP): 被错误地划分为正例的个数，即实际为负例但被分类器划分为正例的实例数

     + False negatives(FN):被错误地划分为负例的个数，即实际为正例但被分类器划分为负例的实例数

     + True negatives(TN): 被正确地划分为负例的个数，即实际为负例且被分类器划分为负例的实例数

     + | Predict\ground-truth |  P   |  N   |
       | :------------------: | :--: | :--: |
       |        **P**         |  TP  |  FP  |
       |        **N**         |  FN  |  TN  |

   + 准确率：被正确分类的样例数目除以所有样例数目
     $$
     \rm{accuracy} = (TP+TN)/(P+N)
     $$

   + 精度：被预测为正例的样本中实际为正的样本数所占比例
     $$
     \rm{precision} = TP/(TP+FP)
     $$

   + 召回率：实际为正例中被预测为正例所占的比例
     $$
     \rm{recall} = TP / (TP + FN)
     $$

   + 敏感度：所有正例中被正确分类的的比例
     $$
     \rm{sensitive} = TP / (TP+FN)
     $$

   + 特异性：所有负例中被正确分类的的比例
     $$
     \rm{specificity}=TN/(TN+FP)
     $$

   + F1-score：精度和召回率的调和平均
     $$
     \rm{F1-score} = \frac{2}{\frac{1}{\rm{precission}}+\frac{1}{\rm{recall}}}
     $$

   对于类别不止两类的情况可以参考这篇[博客](https://blog.csdn.net/sinat_28576553/article/details/80258619)或者[另一篇](https://www.cnblogs.com/bymo/p/8618191.html)

2. 
# bigData
本实验在pyspark环境下进行，通过pycharm进行编程

data文件夹为训练样本集，数据集需要规模大于100行，具体表头内容应为id,features,class的形式

input_data为最终需要选择分类方法的数据集D

代码中的算法集中有三个算法，分别为逻辑回归，朴素贝叶斯，决策树算法。

![image](https://user-images.githubusercontent.com/83856073/147355117-1078b328-6b27-4cf2-89b6-318634dfdb9e.png)

logistic_judge：为分类D所选择的合适算法

write.txt和write1.txt为中间过程文件。

![image](https://user-images.githubusercontent.com/83856073/147355144-5d40909b-f597-4d0f-b2ed-1095ef825189.png)

assusement.py文件表示对数据集D的处理，结果写入write1文件。

代码在master分支里

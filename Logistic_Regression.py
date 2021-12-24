
from pyspark.ml.classification import LogisticRegression

def logistic(train_cv,test_cv):

    lo = LogisticRegression()  ##引入逻辑回归模型
    model1 = lo.fit(train_cv)  ##训练集生成合适的模型
    predictions = model1.transform(test_cv)  ##测试集输出预测值
    print("presiction")
    predictions.show()
    ##统计测试集彼此不同的元素，计算不精确度
    t1 = predictions.select("label").collect()
    t2 = predictions.select("prediction").collect()
    t3 = predictions.count()
    print("t1")
    print(t1)
    t4 = 0
    for i in range(t3):
        if t1[i] == t2[i]:
            t4 += 1
    nbAccuracy = 1.0 * t4 / t3
    print("逻辑回归预测准确值:", nbAccuracy)
    return nbAccuracy

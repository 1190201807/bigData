
from pyspark.ml.classification import LinearSVC

def cal_svm(train_cv,test_cv):
    svm = LinearSVC()  ##引入逻辑回归模型
    # (train_cv, test_cv) = train1.randomSplit([0.7, 0.3])  ##训练集和验证集
    model1 = svm.fit(train_cv)
    print("元数据")
    train_cv.show()
    ##训练集生成合适的模型
    predictions = model1.transform(test_cv)  ##测试集输出预测值
    print("预测值")
    predictions.show()
    predictions.select("label").show(200)
    print("实际值")
    test_cv.select("label").show(200)
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
    print("线性支持向量机预测准确值:", nbAccuracy)
    return nbAccuracy
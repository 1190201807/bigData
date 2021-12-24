
from pyspark.ml.classification import LogisticRegression

def logistic_result(train_cv,test_cv):

    lo = LogisticRegression()
    model1 = lo.fit(train_cv)
    predictions = model1.transform(test_cv)
    print("presiction")
    #predictions.show()
    t2 = predictions.select("prediction").collect()[0]
    print(t2)
    return t2
    ##统计测试集彼此不同的元素，计算不精确度
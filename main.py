import os
from pathlib import Path
from pyspark.ml.feature import PCA as PCAml
from Logistic_Regression import logistic
from assessment import read_input
from dicision_tree import cal_Decision_tree
from svm import cal_svm
from NaiveBayes import cal_NaiveBayes
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import RFormula

##import java.util.UUID

# def mkdir():
#     file_name = './wan.txt'
#     creat_txt(file_name)


if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName("test") \
        .config("spark.some.config.option", "setting") \
        .getOrCreate()

    ##删除中间文件
    my_file = Path("./write.txt")
    if my_file.is_file():
        os.remove(r'./write.txt')

    with open("write.txt", "a") as f:
        for i in range(100):
            f.write('a' + str(i + 1) + ',')
        f.write("class" + '\n')
    f.close()

    ##对于每个数据训练集
    for j in range(16):
        train = spark.read.csv('./data/data' + str(j) + '.txt', header=True, inferSchema=True)
        # test = spark.read.csv('./data.txt', header=True, inferSchema=True)
        # test.printSchema()
        # print(test.head(5))
        # train.describe().show()
        handle = open('data/data' + str(j) + '.txt')
        line = handle.readline()
        print(len(line.split(',')))
        size = len(line.split(','))
        string = line.split(',')[1]  # 特征向量名字
        for i in range(size - 3):
            string = string + "+" + line.split(',')[i + 2]
        print("s")
        print(string)

        ##classfication作为标签
        plan_indexer = StringIndexer(inputCol='classfication', outputCol='class')
        ##我们将fit()方法应用于“train”数据框架上，构建了一个标签。稍后我们将使用这个标签来转换我们的"train"和“test”
        labeller = plan_indexer.fit(train)
        Train1 = labeller.transform(train)
        # Test1 = labeller.transform(test)
        print("Train")
        Train1.show()

        ##一个数据框架的格式，我们需要在这个公式中指定依赖和独立的列；我们还必须为为features列和label列指定名称。
        formula = RFormula(
            formula="class~" + string,
            featuresCol="features", labelCol="label")
        t1 = formula.fit(Train1)
        ##我们需要将这个公式应用到我们的Train1上，并通过这个公式转换Train1
        train1 = t1.transform(Train1)
        # test1 = t1.transform(Test1)
        ##train1.show()##输出拥有feather和label的数据组
        (train_cv, test_cv) = train1.randomSplit([0.7, 0.3])
        lo_accuray = logistic(train_cv, test_cv)
        ##svm_accuray=cal_svm(train_cv,test_cv)
        NaiveBayes_accuray = cal_NaiveBayes(train_cv, test_cv)
        Decision_tree_accuray = cal_Decision_tree(train_cv, test_cv)
        ##所属分类编号
        if (lo_accuray > NaiveBayes_accuray):
            if (Decision_tree_accuray > lo_accuray):
                class_num = 1
            else:
                class_num = 2
        if (lo_accuray <= NaiveBayes_accuray):
            if (Decision_tree_accuray > NaiveBayes_accuray):
                class_num = 1
            else:
                class_num = 0

        print("class_num", class_num)
        print("train_cv show")
        train_cv.select('features').show(40)
        print("Train show")
        train1.show(40)
        print(train_cv.na.drop('any').count())
        train.where("classfication=1").show()
        from pyspark.sql.functions import lit

        train = train.withColumn('classfication', lit(0))
        print("train show")
        train.show()
        # ##col=X.shape[1]
        # X=test[:,:2]
        # ##print(X)
        # Y=test[:,2]
        # X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1)#利用train_test_split进行将训练集和测试集进行分开，test_size占30%
        # print(y_train)#我们看到训练数据的特征值分为3类
        pca = PCAml(k=1, inputCol="features", outputCol="pca")
        model_pca = pca.fit(train1)
        transformed = model_pca.transform(train1)
        print("1")
        print(transformed.show())
        train_final = transformed.select("pca")
        print(train.na.drop('any').count())
        train_final.show()
        t1 = transformed.select("pca").collect()
        t3 = transformed.count()
        # mkdir()
        with open("write.txt", "a") as f:
            f.write(str(t1[0])[21:][:-3] + ',')
            for i in range(99):
                if (i <= t3 - 2):
                    st = str(t1[i + 1])[21:][:-3]
                    f.write(st + ',')
                else:
                    f.write('null' + ',')
            f.write(str(class_num) + '\n')

        f.close()
    test_pca = spark.read.csv('./write.txt', header=True, inferSchema=True)
    handle_final = open('write.txt')
    line = handle_final.readline()
    print(len(line.split(',')))
    size = len(line.split(','))
    string2 = line.split(',')[0]  # 特征向量名字
    for i in range(size - 1):
        string2 = string2 + "+" + line.split(',')[i + 1]
    print("s")
    print(string2)
    formula = RFormula(
        formula="class~" + string2,
        featuresCol="features", labelCol="label")

    t1 = formula.fit(test_pca)
    print("kk1")
    test_pca = t1.transform(test_pca)

    print("222")

    test_pca.show()
    input_data = spark.read.csv('./input_data.txt', header=True, inferSchema=True)
    read_input(input_data)
    pca_final = spark.read.csv('./write1.txt', header=True, inferSchema=True)
    handle_final = open('write1.txt')
    line = handle_final.readline()
    print(len(line.split(',')))
    size = len(line.split(','))
    string1 = line.split(',')[0]  # 特征向量名字
    for i in range(size - 1):
        string1 = string1 + "+" + line.split(',')[i + 1]
    print("s")
    print(string1)

    ##验证数据集的pca的特征值
    formula = RFormula(
        formula="class~" + string1,
        featuresCol="features", labelCol="label")

    t1 = formula.fit(pca_final)
    print("kk")
    pca_final = t1.transform(pca_final)
    pca_final.show()
    from logistic_judge import logistic_result

    method = logistic_result(test_pca, pca_final)
    if str(method) == 'Row(prediction=0.0)':
        print('使用贝叶斯分类器更好')
    if str(method) == 'Row(prediction=1.0)':
        print('使用决策树更好')
    if str(method) == 'Row(prediction=2.0)':
        print('使用逻辑回归更好')

from pyspark.ml.feature import RFormula
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
import os
from pathlib import Path
from pyspark.ml.feature import PCA as PCAml


##评估用哪种方法f
def read_input(input_data):
    ##删除中间文件
    my_file = Path("./write1.txt")
    if my_file.is_file():
        os.remove(r'./write1.txt')

    with open("write1.txt", "a") as f:
        for i in range(99):
            f.write('a'+str(i + 1) + ',')
        f.write('a' + str(200))
        f.write(','+'class')
        f.write('\n')
    f.close()

    handle_final = open('input_data.txt')
    line = handle_final.readline()
    print(len(line.split(',')))
    size = len(line.split(','))
    string_final = line.split(',')[1]  # 特征向量名字
    for i in range(size - 2):
        string_final = string_final + "+" + line.split(',')[i + 2]
    print("s")
    print(string_final)

    ##classfication作为标签
    plan_indexer_final = StringIndexer(inputCol='classfication', outputCol='class')
    ##我们将fit()方法应用于“train”数据框架上，构建了一个标签。稍后我们将使用这个标签来转换我们的"train"和“test”
    labeller = plan_indexer_final.fit(input_data)
    input_data1 = labeller.transform(input_data)
    # Test1 = labeller.transform(test)
    print("input_data1")
    input_data1.show()

    ##一个数据框架的格式，我们需要在这个公式中指定依赖和独立的列；我们还必须为为features列和label列指定名称。
    formula = RFormula(
        formula="class~" + string_final,
        featuresCol="features", labelCol="label")
    t1 = formula.fit(input_data1)
    ##得到带特征值的input_data_final
    input_data_final = t1.transform(input_data1)
    pca = PCAml(k=1, inputCol="features", outputCol="pca")
    model_pca_final = pca.fit(input_data_final)
    transformed_final = model_pca_final.transform(input_data_final)
    print("1")
    print(transformed_final.show())
    transformed_final = transformed_final.select("pca")
    ##print(train.na.drop('any').count())
    transformed_final.show()
    # 中间数据写入文件
    t1 = transformed_final.select("pca").collect()
    t3 = transformed_final.count()
    # mkdir()
    with open("write1.txt", "a") as f:
        # for i in range(t3):
        #     f.write(str(i + 1) + ',')
        # f.write("classfication" + '\n')
        f.write(str(t1[0])[21:][:-3])
        for i in range(99):
            if (i <= t3 - 2):
                st = str(t1[i + 1])[21:][:-3]
                f.write(',' + st)
            else:
                f.write(',' + 'null')
        f.write(','+'null'+'\n')
        for i in range(100):
            f.write(str(i)+',')
        f.write('1')
    f.close()

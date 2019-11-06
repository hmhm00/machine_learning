from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree

'''
使用决策树实现鸢尾花数据集分类
'''
iris=datasets.load_iris()
iris_data=iris.data #特征数据
iris_labels=iris.target#分类类别
print(iris.target)
#train_test_split函数中，test_size代表划分到测试数据集占全部数据集的百分比
data_train,data_test,labels_train,labels_test=train_test_split(iris_data,iris_labels,test_size=0.3)
'''
DecisionTreeClassifier函数参数解析
DecisionTreeClassifier(criterion="gini",  #criterion:特征选择标准,默认是gini(CART),可以设置为entropy(ID3)
                 splitter="best",  #特征划分点选择标准，可选参数，默认是best(更具算法选择最佳的切分特征),
                                   #也可以是random(随机地在局部划分点中找局部最优划分点)
                 max_depth=None,   #决策树最大深度
                 min_samples_split=2, #内部节点再划分所需最小样本数
                 min_samples_leaf=1,  #指定每个样本节点需要的最少样本数
                 min_weight_fraction_leaf=0.,  #指定叶子节点中样本的最小权重
                 max_features=None,  #划分时考虑的最大特征数
                 random_state=None,  #指定了随机数生成器的种子
                 max_leaf_nodes=None, #指定了叶子节点的最大数量
                 min_impurity_decrease=0., #分裂节点判断
                 class_weight=None,
                 presort=False)
'''
#训练决策树
tree_model=tree.DecisionTreeClassifier(max_depth=3)#所有参数均是默认的
tree_model.fit(data_train,labels_train)

#使用模型对测试数据集进行预测
predict_results = tree_model.predict(data_test)
print(predict_results)

#评估模型准确度
result=tree_model.score(data_test,labels_test)
print(result)








'''
功能：该py文件用于测试KNN模型对鸢尾花分类的效果
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

def l1_distance(a,b):
    return np.sqrt(np.sum((a-b)**2,axis=1))
class KNN:
    #初始化K和距离函数
    def __init__(self,n_neighbors=1,dist_funct=l1_distance):
        self.n_neighbors=n_neighbors
        self.dist_funct=dist_funct
    #训练模型方法
    def fit(self,x,y):
        self.x_train=x
        self.y_train=y
    #模型预测方法
    def predict(self,x):
        y_predit=np.zeros((x.shape[0],1),dtype=self.y_train.dtype)
        #遍历输入x的数据点，取出每一个数据点的序号i和数据x_test
        for i,x_test in enumerate(x):
            #x_test和所有训练数据计算距离
            distances=self.dist_funct(self.x_train,x_test)
            #得到的距离按照由近及远排序，取出索引值
            nn_index=np.argsort(distances)
            #选取最近的k个点，保存它们的类别,其中ravel为扁平化多维数组为行向量
            nn_y=self.y_train[nn_index[:self.n_neighbors]].ravel()
            y_predit[i]=np.argmax(np.bincount(nn_y))
            # break
        return y_predit

if __name__ == '__main__':
    knn=KNN(n_neighbors=3)
    knn.fit(x_train,y_train)
    y_predit=knn.predict(x_test)
    accuracy=accuracy_score(y_test,y_predit)
    print("预测准确率",accuracy)


from numpy import *
'''
功能：实现简单的文本分类
'''
class Bayes:
    #创建数据集
    def loadDataSet(self):
        dataSet=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        label=[0,1,0,1,0,1] #1代表侮辱性文档,0代表非侮辱性文档
        return dataSet,label

    #创建所有单词的不重复列表
    def creatVocabList(self,dataSet):
        vocList=[]
        for  document in dataSet:
           for word in document:
               if word not in vocList:
                   vocList.append(word)
        return vocList

    #将文档向量化,
    def setOfWords2Vec(self,vocList,document):
        returnVec=[0]*len(vocList)
        for word in document:
            if word in vocList:
                returnVec[vocList.index(word)]=1
            else:
                print("当前词汇不在该document中"+word)
        return returnVec

    #贝叶斯分类器创建
    def trainNB(self,trainMatrix,trainCatrgory):
        numTrainDocs=len(trainMatrix)#总文档数
        numwords=len(trainMatrix[0])#总词数
        PAbusive=sum(trainCatrgory)/float(numTrainDocs)#侮辱性文档概率
        p0Num=ones(numwords)
        p1Num=ones(numwords)
        p0Denom=2.0
        p1Denom=2.0
        for i in range(numTrainDocs):
            if trainCatrgory[i]==1:
                p1Num+=trainMatrix[i]#数组相加，表示不同的词出现的次数之和
                p1Denom+=sum(trainMatrix[i])#文档出现的总词数
            else:
                p0Num+=trainMatrix[i]
                p0Denom+=sum(trainMatrix[i])
        p1Vect=log(p1Num/p1Denom)#侮辱性文档中每一个词出现的概率
        p0Vect=log(p0Num/p0Denom)#非侮辱性文档中每个词出现的概率
        return p0Vect,p1Vect,PAbusive

    #分类
    def classifyNB(self,vec2Classify, p0Vec, p1Vec, pClass1):
        p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
        p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
        if p1 > p0:
            return 1
        else:
            return 0

    #测试
    def run(self):
        dataSet,label=self.loadDataSet()
        vocList=self.creatVocabList(dataSet)
        trainMat=[]
        for data in dataSet:
            trainMat.append(self.setOfWords2Vec(vocList,data))
        p0V,p1V,pAB=self.trainNB(array(trainMat),array(label))
        print( p0V,p1V,pAB)
        testEntry = ['love','my','dalmation']
        thisDoc = array(self.setOfWords2Vec(vocList,testEntry))
        print(testEntry, 'classified as: ', self.classifyNB(thisDoc, p0V, p1V, pAB))
        testEntry = ['stupid', 'garbage']
        thisDoc = array(self.setOfWords2Vec(vocList, testEntry))
        print(testEntry, 'classified as: ', self.classifyNB(thisDoc, p0V, p1V, pAB))

if __name__ == '__main__':
    bayes=Bayes()
    bayes.run()

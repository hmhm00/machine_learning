from math import log
import operator

#计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for feature in dataSet:
        currentLable=feature[-1]
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable]=0
        labelCounts[currentLable]+=1
    shannonEnt=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

#按照给定特征以及具体的取值划分数据集
def splitDataset(dataSet,axis,value):
    reslt=[]
    for featureVector in dataSet:
        if featureVector[axis]==value:
            feature = featureVector[:axis]
            feature.extend(featureVector[axis+1:])
            reslt.append(feature)
    return reslt

#计算得到最好的数据集时使用的划分数据集的特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 数据集中包含的特征总数
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有的特征
        featList = [example[i] for example in dataSet]  # 该列表包含了整个数据集中当前特征所有取值
        uniqueVals = set(featList)  #去除当前特征取值列表中的重复数据
        newEntropy = 0
        for value in uniqueVals: #遍历不同取值
            subDataSet = splitDataset(dataSet,i,value) #根据当前特征以及取值划分数据集
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        #与初始信息增益进行比较，将信息增益最大的特征赋给bestFeature
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#获取出现最多的类别名称
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回出现类别最多的分类名称
    return sortedClassCount[0][0]

#构建决策树
def createTree(dataset,labels):
    classList=[example[-1] for example in dataset]
    #所有的类别标签完全相同，则直接返回类别标签
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #当数据集中只有一个特征时，以出现次数最多的类别作为返回值
    if len(dataset[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataset)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataset]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        subData=splitDataset(dataset,bestFeat,value)
        myTree[bestFeatLabel][value]=createTree(subData,subLabels)
    return myTree

#创建数据集
def creatDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

if __name__ == '__main__':
    dataset,labels=creatDataSet()
    myTree=createTree(dataset,labels)
    print(myTree)
from numpy import *

def loadDataSet():
    dataMat = [];   labelMat=[]
    fr = open('trainset.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
        labelMat.append(float(lineArr[3]))
    fr.close()
    return dataMat, labelMat

def classify0(inX, dataSet, labels):
    dataSetSize = dataSet.shape[0]
    ####计算欧式距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) ###行向量分别相加，从而得到新的一个行向量
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() ##argsort()根据元素的值从小到大对元素进行排序，返回下标，下标索引
    first = labels[sortedDistIndicies[0]]
    second = labels[sortedDistIndicies[1]]
    third = labels[sortedDistIndicies[2]]
    # final = 0.6*first + 0.6*first * second /(first+second) + 0.2*first * third /(first+third)
    final = first
    return final

def testClassify():
    dataSet, labels = loadDataSet()
    testMat = [];   labelMat = []
    with open('test.txt') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            testMat.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
            labelMat.append(float(lineArr[3]))
    i=0; cal=0
    for inX in testMat:
        # print(inX)
        result = classify0(inX, array(dataSet),labels)
        print("判定结果为：",result ,"误差为：",(result-labelMat[i]))
        cal += (result-labelMat[i])**2
        i = i + 1
    print("总体误差为：",sqrt(cal)/(i-1))


if __name__=='__main__':
    testClassify()

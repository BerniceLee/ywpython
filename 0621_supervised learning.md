# 지도학습 기법 1

### k-NN (k-Nearest Neighbors)

**kNN 기법이란?**

![kNN1](https://t1.daumcdn.net/cfile/tistory/99631D335A165F182D)

	- 물음표에는 세모와 동그라미 중에는 어떤게 들어갈까?, 'A
	- 단순히 물음표 주변에 세모가 가까워서 세모라고 판단하는것이 옳은 판단일까?
	
![kNN2](https://t1.daumcdn.net/cfile/tistory/994A35335A1661A626)

	- 이렇게 보면 파란 동그라미를 넣는게 뭔가 부적절하다고 느껴질것이다.

> kNN은, 단순히 주변에 무엇이 가장 가까이 있는것을 보는게 아니라, 
주변의 몇개의 값을 같이 봐서 가장 많은걸 골라내는 방식이다.
	
##### Euclidean distance

![Euclidean distance](https://wikimedia.org/api/rest_v1/media/math/render/svg/7c75d1876ea130c78887147d389bdfb161fef095)

	- 직교 좌표계에서 두 점 사이의 거리를 계산할 때 흔히 사용함
	- 유클리드 거리 말고 되게 많은 방식으로도 거리 측정을 함
	
* 그 외에도, Manhattan distance, Minkowski distance, Cosine similarity, Mahalanobis distance, Hamming distance 등이 있다.


##### kNN 코드예제1


	- 아래의 코드 예시를 통해서 알아보자

```python
# numpy 에서 *, array
# matplotlib, operator import 해주고

def createDataSet() : 
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]]) # trainingSet
    labels = ['A', 'A', 'B', 'B'] # labels
    return group, labels

group,labels = createDataSet()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(group[:, 0], group[:, 1])
plt.show()
```

	- 4개의 데이터셋을 우선 불러온다.
	- 이를 근간으로 kNN 분류 알고리즘을 짜본다.
	
```python
# kNN 분류 알고리즘 실행
# AttributeError 발생하는데 이걸 해결해보자

# classify0 함수 : 유클리드 거리 구하는 공식을 이용함

def classify0(inX, dataSet, labels, k) :
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet # 각 값들의 차이를 구하고
    sqDiffMat = diffMat**2 # 거기에 제곱을 씌우고
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5 # 값에 루트를 씌워준다 : 유클리드 거리
    sortedDistIndicies = distances.argsort()
    classCount = {}
    
    for i in range(k) : 
        votelabel = labels[sortedDistIndicies[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

classify0([0, 0], group, labels, 3)
```

	- 중간에 AttributeError 뜨는건, 가상환경에서는 3.7 ver. 이고,
	- classCount.itemgetter() 는 3 버전 이상부터는 지원하지 않는다.

```python
classCount.items()
```
	
	- 로 바꿔주면 해결이된다.
	

##### kNN 코드예제2 : 실제 데이터를 이용하여 데이터 사이트의 만남 주선 개선하기


	- 수집 : 제공된 텍스트 파일(datingTestSet.txt)
	- 준비 : 파이썬에서 텍스트 파일 구문 분석하기
	- 분석 : 데이터를 2D 플롯으로 만들기 위해 matplotlib 사용
	- 훈련 : N/A
	- 테스트 : 테스트용 예제 데이터의 일부를 사용하기 위한 함수를 작성
	- 사용 : 사용자가 입력한 몇 가지 데이터를 토대로 누구를 좋아하게 될 것인지를 예측하는데 사용할 수 있는
	- 간단한 커맨드 라인 프로그램을 구축
	
```python
def file2matrix(filename) : 
    love_dictionary = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3)) # 실제 dataSet 사이즈만큼 영벡터 생성
    classLabelVector = [] # 리스트 타입으로 반환
    index = 0
    
    for line in arrayOfLines :
        line = line.strip()
        listFromLine = line.split('\t') # 양끝 탭 공백 없앰
        returnMat[index, :] = listFromLine[0:3]
        # classLabelVector.append(listFromLine[-1])
        if (listFromLine[-1].isdigit()) :
            # 해당 부분 코드는 협업을 위한 코드 유연성 문제로 이렇게 코드를 친것
            classLabelVector.append(int(listFromLine[-1]))
        else :
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index+=1
        
    return returnMat, classLabelVector

datingDataMat, datingLabels = file2matrix('C:/Users/Affinity/Desktop/개인자료/module3/datingTestSet.txt')
datingDataMat
datingLabels[0:20]
```

	- 이걸 시각화 해보자
	
```python
# 시각화

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
ax1 = fig.add_subplot(122)
ax1.scatter(datingDataMat[:, -1], datingDataMat[:, 2],
           15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()
```

  - 시각화는 했는데, 그 다음 단계는?
  - 각 수치형 자료를 정규화 해보자.
  
 > 값이 서로 다른 범위에 놓여 있는 경우, 이를 정규화해준 이후 사용해야 한다.
 
  - 일반적으로, 정규화는 0~1 또는 -1~1 범위로 지정
    - 0~1 범위로 정규화 하고자 할 땐, 다음과 같은 공식을 적용한다.
    
 ```python
 newValue = (oldVaule - min) / (max - min)
 ```
 
  - 해당 공식을 근간으로 정규화를 해보자.
  
 ```python
 # 수치형 데이터 정규화

def autoNorm(dataSet) : 
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) # newValue 들이 들어갈 영행렬
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1) # oldValue - min
    normDataSet = normDataSet / tile(ranges, (m,1)) # max - min으로 나눠줌
    return normDataSet, ranges, minVals

normMat, ranges, minVals = autoNorm(datingDataMat)
normMat
ranges
minVals
```


  - 정규화를 하게되면
    -  array([[0.44832535, 0.39805139, 0.56233353],
    
       [0.15873259, 0.34195467, 0.98724416],
       
       [0.28542943, 0.06892523, 0.47449629],
       ...,
       
       [0.29115949, 0.50910294, 0.51079493],
       
       [0.52711097, 0.43665451, 0.4290048 ],
       
       [0.47940793, 0.3768091 , 0.78571804]])
  - 이런 식으로 0~1 사이 값들로 정규화가 된다.
  
  - 정규화가 된 데이터들로 실제 분류기를 테스트 진행해보자.
  
```python
# 전체 프로그램으로 분류기 테스트

def datingClassTest() :
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('C:/Users/Affinity/Desktop/개인자료/module3/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    
    for i in range(numTestVecs) :
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                    datingLabels[numTestVecs:m], 3)
        print("The classifier came back with : {}, the real abswer is: {}".format(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]) :
            errorCount += 1.0
    
    print("The total error rate is : {}".format(errorCount / float(numTestVecs)))
    print("The error count : {}".format(errorCount))
    
datingClassTest()
```

  - 성공적으로 분류기가 테스팅 되면,
    - The classifier came back with : 2, the real abswer is: 2
    
      The classifier came back with : 1, the real abswer is: 1
      
      The classifier came back with : 3, the real abswer is: 1
      
      The total error rate is : 0.05
      
      The error count : 5.0
      
  - 이런 식의 테스트 결과를 얻을 수 있다.
  
  - 테스트가 끝났으면, 실제로 써먹을 시스템을 만들자.
  
 ```python
 def classifyPerson() :
    resultList = ['not at all', 'in small doses', 'in large doses']
    print("모든 데이터는 숫자로만 입력해주세요\n")
    percentTats = float(input("비디오 게임으로 보내는 시간의 비율 (0~100% 사이로 입력) : "))
    ffMiles = float(input("연간 항공 마일리지 수 : "))
    iceCream = float(input("연간 아이스크림 소비량 (L 단위로 입력 / 예 : 0.5L) : "))
    datingDataMat, dtingLabels = file2matrix('C:/Users/Affinity/Desktop/개인자료/module3/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person : {}".format(resultList[classifierResult - 1]))
    
classifyPerson()
```

  - 실행시키고, 3가지 항목에 대한 답을 입력하면
    - 모든 데이터는 숫자로만 입력해주세요

      비디오 게임으로 보내는 시간의 비율 (0~100% 사이로 입력) : 35
      
      연간 항공 마일리지 수 : 100000
      
      연간 아이스크림 소비량 (L 단위로 입력 / 예 : 0.5L) : 3
      
      You will probably like this person : not at all
      
   - 이런 형식의 최종 예측값을 얻을 수 있다.
   
 
 ### Decision Tree
 
 
 ##### Decision Tree 란?
 
	- 분류와 회귀 문제에 널리 사용함
	- 각 변수들의 속성에 따라 구분 선을 긋고, 기준선에 미달이면 결과1, 이상이면 결과2, 이런식으로 구분함
	
	- 스무고개와 비슷하다.
	
![decision tree1](https://tensorflowkorea.files.wordpress.com/2017/06/2-22.png?w=768&h=546)

	- 맨 처음이 뿌리 노드, 가지들을 지나 맨 마지막 (leaf) 까지 도달한다.
	- 실제로 결정트리를 만들때, 어떤 데이터 셋에 두 개의 변수가 있다고 가정하면
	
![decision tree2](https://tensorflowkorea.files.wordpress.com/2017/06/2-23.png?w=768)

	- 이걸 특정한 기준선으로 나누면
	
![decision tree3](https://tensorflowkorea.files.wordpress.com/2017/06/2-24.png?w=768)

	- 기준선을 긋는 단계(깊이) 가 깊어지면
	
![decision tree4](https://tensorflowkorea.files.wordpress.com/2017/06/2-25.png?w=768)

	- 이런 식으로 깊이가 깊어지면서 세밀하게 분류가 가능함.
	

> 아니 그럼 대체 분류선은 어떻게 그어야 하는건가?

	- 꽃잎으로 꽃을 판단해주는 decision tree 가 있다고 해보자.
	
![decision tree_leafs](https://t1.daumcdn.net/cfile/tistory/99CDA3345B580E0320)

	- 분류를 했을 때, 순종인지 잡종인지 구분을 우선 해준다.
		- 즉, 잘 분류한건지 잘못 분류한건지 체크한다.
		
	- 이 작업을 눈대중으로 할 순 없으니, 엔트로피를 계산하다.
	
**엔트로피란?**

(참고 URL : https://ko.wikipedia.org/wiki/%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC)

	- 통계학에선 엔트로피의 차, 절대적 값을 정의할 수 있다.
	- 어떤 자료의 혼잡도 등을 체크하고자 할 때 쓴다.
	- 확률적 상태 분포를 가지는 어떤 계의 앙상블을 생각하자. 
		- 여기서 단일계의 상태(미시적 상태) {\displaystyle i} i의 확률을 {\displaystyle p_{i}} p_i라고 하자. 이 경우, 앙상블의 엔트로피 S는 다음과 같이 정의한다.
		
![entropy](https://wikimedia.org/api/rest_v1/media/math/render/svg/45a5b959edf49a7de8f2965318b11c3c4b7b38b1)

	
	- 보통 0~1 사이의 값을 가짐
	- 이 공식을 사용하여 순종성(엔트로프)의 변화를 계산하려면,
	- 엔트로피1 (쪼개기 전), 엔트로피2 (쪼갠 후)의 차이를 구해야한다.
		
> InfoGain(F) = Entropy(S1) - Entropy(S2)

	- 그런데, 여러번 선을 그으며 분할하기 때문에, 분할할때마다 엔트로피에 가중치를 부여해야 한다.
	

**Information Gain**

	- 데이터를 분할하기 전(상위)과 분할 후(하위)의 변화
	
	- "어떤 속성으로 데이터를 분할할 때 가장 높은 정보 이득을 취할 수 있을까?"
	- 정보 이득이 가장 높은 속성을 가지고 분할하는 것이 가장 좋은 선택
	
	- 그런데, information Gain은 outcome이 많으면 bias 되는 문제가 있다.
		- 즉, 많은 측정 값을 가진  속성으로 편향된다.

**Gini Impurity**

	- 위의 Information Gain 말고도 지니계수를 쓰기도 함.
	- CART 알고리즘에서 cost func 구할때도 함
	
어쨌든, Entrophy를 이용한 Information Gain 과 Gini Impurity 둘다 쓰는데,

	- Information Gain : 엔트로피가 로그 연산이라 조금 느리다. 하지만 좀 더 balanced 된 트리 생성 가능
	- Gini Impurity : 연산 속도가 좀 더 빠르지만, 확 나눠버리는 특성이 있음.
	

##### Decision Tree 예제코드1

	- 바다에서 얻을 수 있는 다섯 종류의 동물이 지표면에 닿지 않고 살아남을 수 있는지, 지느러미는 있는지 묻는 내용
	- 이 동물들은 두 가지 항목으로 분류 (물고기다/아니다)
	- No surfacing 으로 분류할지 filppers로 분류할지
	
	
```python
from math import log
import operator

def createDataSet() :
    dataSet = [[1, 1, 'yes'],
              [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannonEnt(dataSet) :
    numEntries = len(dataSet)
    labelCounts = {}
    
    for featVec in dataSet :
        currentLabel = featVec[-1] # 맨 끝의 값을 현재라벨로 정의
        if currentLabel not in labelCounts.keys() :
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    
    for key in labelCounts : # labelCounts = {'yes':2, 'no':3}
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2) # 엔트리피 구하는 공식으로 엔트리피 산출
    return shannonEnt

myDat, labels = createDataSet()
calcShannonEnt(myDat)
```

	- 0.9709505944546686 라는 엔트리피 값을 얻음
	
```python
myDat[0][-1] = 'maybe'
calcShannonEnt(myDat)
```
	
	- maybe 라는 값을 넣고 계산하면 1.3709505944546687 엔트리피 값이 나옴
	
	- 엔트리피를 계산했으면, 데이터 셋을 분해해보자.
		- 여러 속성 중, 정보 이득이 가장 큰 속성을 가지고 데이터셋 분할
		- 분류 알고리즘을 작동하려면?
			- 엔트로피를 측정하고 데이터셋을 분할하고,
			- 분할된 셋의 분할된 셋의 엔트로피를 측정하고
			- 분할이 올바르게 되엇는지 확인해야함
		- 데이터셋 분할에 가장 좋은 속성을 결정하려면, 모든 속성을 가지고 엔트로피를 구해봐야함
		
```python
# 데이터셋 분할

def splitDataSet(dataSet, axis, value) : 
    retDataSet = []
    for featVec in dataSet :
        if featVec[axis] == value :
            reduceFeatVec = featVec[:axis] # 값 분류로 쓰여진 축은 제외시킨다
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

splitDataSet(myDat, 0, 0)
```

	- splitDataSet(myDat, 0, 0) : [[1, 'no'], [1, 'no']]
	- splitDataSet(myDat, 0, 1) : [[1, 'maybe'], [1, 'yes'], [0, 'no']]
	- splitDataSet(myDat, 1, 0) : [[1, 'no']]
	
	- 엔트로피를 구하기 위해 데이터셋을 미리 축에 따라 쪼개준다.
	- 이 기능을 하는 함수를 미리 구현한 것.
	
	- 데이터 분할 기능이 생겼으면, 분할할 때 가장 좋은 속성을 선택해보자.
	
	
```python
# 데이터셋 분할시 가장 좋은 속성 선택하기

def chooseBestFeatureToSplit(dataSet) :
    numFeatures = len(dataSet[0]) - 1 # 맨 마지막 열은 라벨을 위해 쓸거라서 하나 뺀다.
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    
    for i in range(numFeatures) :  
        featList = [example[i] for example in dataSet] # 모든 예에 대한 list 생성
        uniqueVals = set(featList) # 리스트에 대한 집합 생성
        newEntropy = 0.0
        
        for value in uniqueVals :
            subDataSet = splitDataSet(dataSet, i, value) # value 값을 집합 원소에 따라 서브 셋에 분류해서 넣고,
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        InfoGain = baseEntropy - newEntropy # 상위 엔트로피에서 하위 엔트로피를 빼서 InfoGain 얻기
        
        if (InfoGain > bestInfoGain) :
            bestInfoGain = InfoGain # 얻어진 InfoGain 을 best 값과 비교하며, 더 나으면 계속 대체함
            bestFeature = i
            
    return bestFeature

chooseBestFeatureToSplit(myDat)
```

	- bestEntropy 는 0으로 나온다. (값을 변화시키면 엔트로피도 바뀜)
	
	- bestEntropy 도 구했으면, 재귀적으로 결정 트리를 만들어본다.
		- 여기서, 제시한 알고리즘이 더이상 분할할 속성이 없거나, 하나의 가지에 있는 모든 사례가 전부 같은 분류 항목일때 멈추게 해야한다.
		- 여기선 속성을 다 썼는지, 어떤지를 확인하기 위해 데이터셋에 있는 컬럼을 간단하게 셈
		- 만약에 데이터셋에 있는 속성을 다 썼는데도 분류 항목의 표시 개수가 같지 않으면,
			- leaf node를 불러야 할지를 결정해야함
		- 이 때 다수결(majority vote)을 사용한다.
		- majorityCnt 함수는 분류 항목명의 목록을 가져와서 classList 내에 유일한 key값이 되도록 딕셔너리 생성
		- 이 딕셔너리의 대상은 classList에서 각각의 분류 항목에 대한 발생 빈도이다.
		- 마지막으로, 딕셔너리를 정렬하고 발생 빈도가 가장 큰 분류 항목을 반환해준다.
		
		
```python
# 분류 항목의 목록을 가져옴

def majorityCnt(classList) :
    classCount = {}
    for vote in classList :
        if vote not in classCount.keys() :
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
```

```python
# 재귀적으로 트리 생성

def createTree(dataSet, labels) : 
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0])== len(classList) : 
        return classList[0] # 첫번째 멈춤 조건 : 모든 분류 항목이 같을 때 멈춘다.
    if len(dataSet[0]) == 1 :
        return majorityCnt(classList) # 두번째 멈춤 조건 : 더이상 분류할 속성이 없을 때, 가장 많은 속성을 반환
    
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel : {}} # 유일한 값의 리스트를 구하고 딕셔너리에 넣음
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    
    for value in uniqueVals :
        subLabels = labels[:] # 모든 값의 라벨들을 서브라벨에 복사함
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
```

	- 아직은 트리에 값이 0, 1로만 되어있어서, 실제로 분류하고자 하는 no surfacing 이랑 flippers로 트리 바꿔준다.
	
```python
def retriveTree(i) :
    listOfTrees = [{'no surfacing' : {0: 'no',
                                     1: {'flippers' : {0: 'no',
                                                      1: 'yes'}}}},
                  {'no surfacing' : {0: 'no',
                                    1: {'flippers' : {0: {'head' : {0: 'no',
                                                                   1: 'yes'}},
                                                     1: 'no'}}}}]
    return listOfTrees[i]
```

```python
def classify1(inputTree, featLabels, testVec) :
    # retriveTree 함수를 통해 생성되는 트리의 최종 데이터형이 list기 때문에
    # inputTree 로 하나, inputTree.keys()로 하나 똑같이 출력되지만
    # 트리의 최종 자료형이 달라지는 경우에는 keys() 나 values()를 통해서 추출해서 쓰거나 한다.
    firstStr = list(inputTree)[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    
    if isinstance(valueOfFeat, dict) :
        classLabel = classify1(valueOfFeat, featLabels, testVec)
    else :
        classLabel = valueOfFeat
    return classLabel

myTree = retriveTree(0)
classify1(myTree, labels, [1,0])
```

	- 이렇게 하면 'no'로 출력된다.
	
```python
classify1(myTree, labels, [1,1])
```
	
	- 이렇게 하면 'yes'로 출력된다.
	
	

### Linear Regression


##### 선형회귀란?

![linear regression](https://user-images.githubusercontent.com/31824102/35181510-dc9c2998-fdba-11e7-9a2a-edf38c40b3f3.PNG)

	- 독립변수와 종속변수의 관계가 선형으로 나오는 경우
	
```python
# 선형회귀 

np.random.seed(seed=1)
X_min = 4
X_max = 30
X_n = 16
X = 5 + 25 * np.random.rand(X_n)
Prm_c = [170 ,108, 0.2]
T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) + 4 * np.random.randn(X_n)
np.savez('ch5_data.npz', X=X, X_min=X_min, X_max=X_max, X_n=X_n, T=T)
print(np.round(X, 2))
print(np.round(T, 2))

plt.figure(figsize=(4,4))
plt.plot(X, T, marker='o', linestyle='None', markeredgecolor='black', color='cornflowerblue')
plt.xlim(X_min, X_max)
plt.grid(True)
plt.show()
```

	- 출력결과는 ipynb에서
	- 정확히 맞추는건 불가하고, 어느정도 오차를 허옹하면 그럴듯하게 직선 그려서 예측 가능


**평균 제곱 오차(mean square error, MSE) 함수

![mse](https://wikimedia.org/api/rest_v1/media/math/render/svg/67b9ac7353c6a2710e35180238efe54faf4d9c15)


	- 오차의 제곱에 대해 평균을 취함
	- MSE가 작을 수록 원본과의 오차가 적음
	
	- w0, w1을 결정하면 그에 대한 평균 제곱 오차를 계산할 수 있다.
	- 그러나, 어떤 w0과 w1을 선택하더라도, 데이터가 직선 상에 나란히 있지 않으므로 MSE가 완전히 0은 안됨
	
	- 아래 예제를 통해 w와 MSE와의 관계를 살펴보자
	
	
```python
# w와 MSE와의 관계

from mpl_toolkits.mplot3d import Axes3D

def mse_line(x, t, w) :
    y = w[0] * x + w[1]
    mse = np.mean((y - t)**2)
    return mse

xn = 100
w0_range = [-25, 25]
w1_range = [120, 170]
x0 = np.linspace(w0_range[0], w0_range[1], xn)
x1 = np.linspace(w1_range[0], w1_range[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
MSE = np.zeros((len(x0), len(x1)))

for i0 in range(xn) :
    for i1 in range(xn) :
        MSE[i1, i0] = mse_line(X, T, (x0[i0], x1[i1]))
        
plt.figure(figsize=(9.5, 4))
plt.subplots_adjust(wspace=0.5)

ax = plt.subplot(1, 2, 1, projection='3d')
ax.plot_surface(xx0, xx1, MSE, rstride=10, cstride=10, alpha=0.3, color='blue', edgecolor='black')
ax.set_xticks([-20, 0, 20])
ax.set_yticks([120, 140, 160])
ax.view_init(20, -60)

plt.subplot(1, 2, 2)
cont = plt.contour(xx0, xx1, MSE, 30, colors='black', levels=[100, 1000, 10000, 100000])
cont.clabel(fmt='%1.0f', fontsize=8)
plt.grid(True)
plt.show()
```

	- 출력 결과는 ipynb
	- 왼쪽 그래프는 MSE, 오른쪽 그래프는 MSE의 등고선 표시
		- MSE 보면 w0 방향의 변화에 MSE가 크게 변한다.
			- 기울기가 조금이라도 바뀌면, 직선이 데이터 점에서 크게 어긋나기 때문이다.
			- 그러나, 3D 그래프는 w1 방향의 변화를 알기엔 어렵다.
		- 오른쪽 그림은 w0 = 3, w1 = 135 근처에서 MSE가 최소값을 취할 것 같다.
		
	- MSE도 구했으면, 매개 변수를 구해본다.
	
	
**경사하강법**


> MSE가 가장 작아지는 w0과 w1은 어떻게 구할까?

	- 우선 초기 위치로 적당한 w0과 w1을 결정
	- 이것은 MSE 지형 위의 한 지점에 대응함
	- 이 점에서의 기울기를 확인하고, MSE가 가장 감소하는 방향으로 w0과 w1을 조금만 진행
	- 이 절차를 여러번 반복
	- 최종적으로 MSE가 가장 작아지는 그릇의 바닥인 w0과 w1에 도달 가능
	
	- 어느 지점(w0, w1)에 서서 주위를 빙 둘러봤을 때
	- 언덕의 위쪽 방향은 MSE를 w0과 w1로 편미분한 벡터를 표시
	- 이것을 MSE의 기울기로 부름
	- MSE를 최소화 하려면 MSE의 기울기의 반대 방향으로 진행
	
	
```python
# w0, w1의 기울기 구하기

def dmse_line(x, t, w) :
    y = w[0]* x + w[1]
    d_w0 = 2 * np.mean((y - t) * x)
    d_w1 = 2 * np.mean(y - t)
    return d_w0, d_w1

d_w = dmse_line(X, T, [10, 165])
print(np.round(d_w, 1))
```

	- [5046.3  301.8] 로 나옴. w0과 w1의 기울기이며, w0의 기울기가 w1 기울기보다 크다.
	
	- 매개변수를 구한다. 여기에는 경사하강법을 이용한다.
	- 경사하강법 함수를 구현해보자 (fit_line_num(x,t))
		- fit_line_num(x,t) 는 데이터 x,t를 인수로하여 mse_line을 최소화 하는 w를 반환한다.
		- w는 초기값 w_init = [10.0, 165.0] 에서 시작, dmse_line에서 구한 기울기 w를 갱신한다.
		- 갱신 단계의 폭이 되는 학습 비율은 alpha = 0.001
		- w이 평평한 곳에 도달하면(즉, 기울기가 충분하게 적어지면) w의 갱신을 종료
			- 즉, 기울기의 값이 eps = 0.1보다 작아지면 빠져나옴
		- 프로그램 실행시 마지막으로 얻어진 w값, w의 갱신내역 등을 표시함
		
		
```python
# 경사하강법

def fit_lie_num(x, t) :
    w_init = [10.0, 165.0] # 초기 매개 변수
    alpha = 0.001 # 학습률
    i_max = 100000 # 반복 최대수
    eps = 0.1 # 기울기 절대값의 한계
    w_i = np.zeros([i_max, 2])
    w_i[0, :] = w_init
    
    for i in range(1, i_max) :
        dmse = dmse_line(x, t, w_i[i -1])
        w_i[i, 0] = w_i[i - 1, 0] - alpha * dmse[0]
        w_i[i, 1] = w_i[i - 1, 1] - alpha * dmse[1]
        
        if max(np.absolute(dmse)) < eps : 
            break # 절대값보다 작아지면 종료 판정
    w0 = w_i[i, 0]
    w1 = w_i[i, 1]
    w_i = w_i[:i, :]
    return w0, w1, dmse, w_i
    
plt.figure(figsize=(4,4)) # MSE 등고선 표시
xn = 100 # 등고선 해상도
w0_range = [-25, 25]
w1_range = [120, 170]
x0 = np.linspace(w0_range[0], w0_range[1], xn)
x1 = np.linspace(w1_range[0], w1_range[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)

for i0 in range(xn) :
    for i1 in range(xn) :
        MSE[i1, i0] = mse_line(X, T, (x0[i0], x1[i1]))

cont = plt.contour(xx0, xx1, MSE, 30, colors='black', levels=[100, 1000, 10000, 100000])
cont.clabel(fmt='%1.0f', fontsize=8)
plt.grid(True)
plt.show()
```

	- 출력 결과는 ipynb에서..
	- 등고선에 w값이 뜨기는 떴는데.. 뭐가 갱신값인지 잘 모르겠음
	
```python
# w 갱신값 표시하여 재출력

plt.figure(figsize=(4,4)) # MSE 등고선 표시
xn = 100 # 등고선 해상도
w0_range = [-25, 25]
w1_range = [120, 170]
x0 = np.linspace(w0_range[0], w0_range[1], xn)
x1 = np.linspace(w1_range[0], w1_range[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)

for i0 in range(xn) :
    for i1 in range(xn) :
        MSE[i1, i0] = mse_line(X, T, (x0[i0], x1[i1]))

cont = plt.contour(xx0, xx1, MSE, 30, colors='black', levels=[100, 1000, 10000, 100000])
cont.clabel(fmt='%1.0f', fontsize=8)
plt.grid(True)

# 이 부분의 코드를 추가한다.
# 경사 하강법 호출
W0, W1, dMSE, W_history = fit_line_num(X, T)

# 갱신값 포함하여 결과 보기
print("반복 횟수 {0}\n".format(W_history.shape[0]))
print("W = [{0:.6f}, {1:.6f}]\n".format(W0, W1))
print('dMSE = [{0:.6f}, {1:.6f}]\n'.format(dMSE[0], dMSE[1]))
print('MSE = {0:.6f}\n'.format(mse_line(X, T, [W0, W1])))
plt.plot(W_history[:, 0], W_history[:, 1], '.-',
        color='gray', markersize=10, markeredgecolor='cornflowerblue')
plt.show()
```

	- 출력 결과는 ipynb
	- 파랗게 표기된 부분이 w의 갱신
	- 처음에는 가파른 계곡으로 진행해 골짜기에 정착하면, 계곡의 중앙 부근에 천천히 나아가서 기울기가 거의 없어지는 지점에 도달
	
	- w0, w1 아까 구했던거를 직선 식에 대입해본다.
	
	
```python
# w0, w1 직선식에 대입하기

# 선 표시하기
def show_line(w) :
    xb = np.linspace(X_min, X_max, 100)
    y = w[0] * xb + w[1]
    plt.plot(xb, y, color=(.5, .5, .5), linewidth=4)
    
plt.figure(figsize=(4,4))
W = np.array([W0, W1])
mse = mse_line(X, T, W)
print("w0 = [0:.3f], w1 = {1:.3f}\n".format(W0, W1))
print("SD = {0:.3f} cm\n".format(np.sqrt(mse)))
show_line(W)

plt.plot(X, T, marker='o', linestyle='None', color='cornflowerblue', markeredgecolor='black')
plt.xlim(X_min, X_max)
plt.grid(True)
plt.show()
```

	- 출력 결과는 ipynb
	- 출력 결과를 보면, w0 = 1.540, w1 = 136.176, SD = 7.002 cm 이렇게 나온다.
	
	- 이를 통해서 회귀방정식을 구해보면
		- y = (w0)x + (w1)
		- y = 1.5x + 136.175

> 데이터가 근데 완전히 일치하지가 않는다. 그럼 데이터랑 얼마나 차이가 날까?

	- 평균 제곱 오차가 49.03cm² (직관적이지 않다)
	- 평균 제곱 오차의 제곱근 = 7.00cm (표준편차)
	
	- 오차가 대략 7.00cm 라는거는.....
		- 오차가 정규분포를 따른다고 가정 했을 때, 전체의 68%의 데이터 점에서 오차가 7cm 이하라는 뜻
		- 정규 분포의 경우, 평균에서의 차이가 표준편차의 범위에 분포의 68%가 들어감
		

	
### SVM (Support Vector Machine)

##### SVM 이란?

	- 퍼셉트론 : 가장 단순하고 빠른 판별 함수 기반 분류 모형이지만 판별 경계선(decision hyperplane)이 유니크하게 존재하지 않는다는 특징
	- SVM은 퍼셉트론을 기반으로, 가장 안정적인 판별 경계선을 찾기 위하 제한 조건을 추가한 모형
	
![SVM1](https://datascienceschool.net/upfiles/3e0a11567a154fc6a5e25010a68b5872.png)

> "How do we divide the space with decision boundaries?"

	- 예시를 통해 살펴보자
	
![SVM ex1](https://4.bp.blogspot.com/-ln7nyPwA7UE/WmwCQd-IeVI/AAAAAAAACfI/Cus1PN4tAuMUgTUqELM6yWYDBoCQeC1CgCK4BGAYYCw/s200/svm1.png)

	- 이런 이슈들을 생각해볼 수 있다.
	
	1. 우리가 ′+′ 샘플과 ′−′ 샘플을 구별하고 싶다면 어떤 식으로 나눠야 하는가? 
	2. 만약 선을 그어 그 사이를 나눈다면 어떤 선이어야 할 것인가? 
	
	- 아마도 ′+′와 ′−′ 샘플 사이의 거리를 가장 넓게 쓰는 어떤 line으로 다음과 같은 녀석(점선)일 것이다.
	
![SVM ex2](https://4.bp.blogspot.com/-pYY6U0peR74/WmwCRbFsM7I/AAAAAAAACfQ/gwcY8_j13yU6qwkITBmgd2yrVkKd2hvQwCK4BGAYYCw/s200/svm2.png)

	- 직관적으로도 이렇게 풀 수 있지만, SVM은 체계적으로 아이디어를 개발하고 논리를 전개하는 과정 그 자체다.
	- 과정들을 이제 살펴보자
	
	
**Decision Rule**


	- w⃗  를 하나 그려보자. 이 벡터는 우리가 그릴 street의 중심선에 대해 직교하는 벡터다.
	
![SVM ex3](https://1.bp.blogspot.com/-n2tSWYD_XoA/WmwE1cP5ctI/AAAAAAAACfo/3eniT7SzJioB4SdJVL-HTJSb_ouTLLQ_gCK4BGAYYCw/s200/svm3.png)

	- 그리고 이제 모르는 샘플 u⃗  하나가 있을 때 우리가 궁금한 것은 
	- street를 기준으로 이 녀석이 오른쪽에 속할지 혹은 왼쪽에 속할지이다.
	
		- 두 벡터를 내적해보고, 그 값이 상수 c보다 큰지 안큰지 한번 비교해보는 것이다.
	
	- 따라서, w⃗ ⋅u⃗ +b≥0  then ‘+', 이 내적식이 decision rule 이 된다.
	
		- 하지만, 저 수식에서 어떤 벡터 w를 잡아야 하는지, 어떤 b를 잡아야 하는지 우린 아무것도 모름.
		- 이걸 계산할 수 있도록, 여러 제약 조건들을 추가해보는 작업을 해보자
		
		
**Design and add Additional constrains**

	- 위의 식에서 조금 더 나아가서 x+를 ‘+′ 샘플 x−가 ‘−′ 샘플이라 할 때 다음과 같이 적어보자
	
		- w⃗ ⋅x⃗ ++b≥1
		- w⃗ ⋅x⃗ −+b≤−1
	
	- 즉, `+' 샘플을 예를 들면 이 샘플에 대해서는 우리의 decision rule이 최소한 1보다는 큰 값을 주도록 해본 것
	
	
> 구체화 되긴 했는데, 여전히 문제가 쉬워지지는 않았다?

	- 여기에 variable 하나를 고안해서 문제를 좀 바꿔보자.
	
		- yi={ 1 for‘+′ / -1 for‘−′ } 
	
	- 이제 이 새로운 variable yi 를 수식에 각각 곱한다.
	
		- yi(w⃗ ⋅x⃗ i+b)≥1
		- yi(w⃗ ⋅x⃗ i+b)−1≥0 (위에꺼를 조금 정리한 수식)
	
	- 여기까지 했을 때, 등호가 성립할 때는 x⃗ i가 정확히 street의 양 쪽 노란 경계선에 정확히 걸쳐 있을 때라는 제약을 하나 더 추가해보자.
	
	- 즉, 위에 그림에서 경계(노란선, gutters)에 걸칠 ‘+′ 샘플 하나와 ‘−′ 샘플 두 개에 대한 수식의 결과가 0이 된다.
	
		- yi(w⃗ ⋅x⃗ i+b)−1=0forx⃗ i∈노란선 (gutters)
		
		
	- 이 쯤에서 우리가 하고자 했던 목적을 상기해보자.
	- 우리는  ‘+′ 샘플과 ‘−′ 샘플 사이의 거리를 가능한 최대로 넓게 하고 싶다.
	
![SVM ex4](https://2.bp.blogspot.com/-Kb58kvRn3p0/WmwKaCgfcQI/AAAAAAAACgA/FprZTNJDE7cLr_37_7AvnPuLe0PBuW9hgCK4BGAYYCw/s200/svm4.png)

	- WIDTH 거리값은 결국, 2 / w벡터의 절대값이 된다.
	
	
> SVM 모델은, WIDTH 거리를 최대화 하고싶다는 목적을 가지고 있다!

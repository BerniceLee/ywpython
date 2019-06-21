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
		
![entrophy](https://wikimedia.org/api/rest_v1/media/math/render/svg/45a5b959edf49a7de8f2965318b11c3c4b7b38b1)

	
	- 보통 0~1 사이의 값을 가짐
	- 이 공식을 사용하여 순종성(엔트로프)의 변화를 계산하려면,
	- 엔트로피1 (쪼개기 전), 엔트로피2 (쪼갠 후)의 차이를 구해야한다.
		
> InfoGain(F) = Entrophy(S1) - Entrophy(S2)

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
	

**규제 매개변수(
	

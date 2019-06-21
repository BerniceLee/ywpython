# 지도학습 기법 1

### k-NN (k-Nearest Neighbors)

**kNN 기법이란?**

![kNN1](https://t1.daumcdn.net/cfile/tistory/99631D335A165F182D)

	- 물음표에는 세모와 동그라미 중에는 어떤게 들어갈까?, 'A
	- 단순히 물음표 주변에 세모가 가까워서 세모라고 판단하는것이 옳은 판단일까?
	
![kNN2](https://t1.daumcdn.net/cfile/tistory/994A35335A1661A626]

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
	python
'''python
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

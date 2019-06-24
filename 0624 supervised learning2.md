# 지도학습 기법2

### Naive Bayes

##### Bayes Theory

네이브 베이즈 이론을 보기위한 베이즈 이론

![조건부확률](https://t1.daumcdn.net/cfile/tistory/99BBBA335A1CE3AB20)

  - 조건부 확률 P(A|B)는 사건 B가 발생한 경우 A의 확률을 나타냄
  
 > 베이즈 정리 : P(A|B) 의 추정이 P(A∩B) 와 P(B) 에 기반을 두어야한다는 정리
 
  - 아래의 예시를 통해 살펴보자
  
![bayes example1](https://t1.daumcdn.net/cfile/tistory/9927EC335A1CE9C01B)
 
  - 전체 사건 중 비가 온 확률은 P(비) = 7/20 이다. 그렇다면 비가 안온 확률은 얼마일까?
  - P(~비) = 13/20. 비가 오는지 안오는지 같이 둘 중 하나의 상태만 가능한 사건들은 모든 경우의 수를 더했을 때 1이 됨.
  
  - 그럼 이제 P(비|맑은날)의 값은 얼마?
  
![조건부확률2](https://t1.daumcdn.net/cfile/tistory/999A41335A1CE91C1C)
 
  - 위 식을 통해 P(비|맑은날) 을 구하기 위해선 P(맑은날|비), P(비), P(맑은날) 이 세개의 값만 알아내면 됨.
  
![bayes example2](https://t1.daumcdn.net/cfile/tistory/99F2A3335A1CF96304)

  - P(비|맑은날) = P(맑은날|비) * P(비) / P(맑은날)
  - = (2/7) * 0.35 / 0.5 = 0.2
  
  - 전체중에서 맑은날 이면서 비가올 확률은 20% 정도 된다고 볼 수 있다.
  
**나이브 베이즈 공식**
 
![조건부확률2](https://t1.daumcdn.net/cfile/tistory/999A41335A1CE91C1C)

  - 데이터 B의 관점에서 가설 A의 확률을 추정한다.
  
  - P(A) : 사전 확률, 데이터를 보기 전의 가설 확률
  - P(A|B) : 사후 확률, 데이터를 확인한 이후의 가설 확률
  - P(B|A) : 우도(조건부 확률), 데이터가 가설에 포함될 확률
  - P(B) : 한정한수(evidence), 어떤 가설에도 포함되는 데이터의 비율
  
  
**ML ＆ MAP**

(참고 URL : https://m.blog.naver.com/PostView.nhn?blogId=dgmktg&logNo=220713082304&proxyReferer=https%3A%2F%2Fwww.google.com%2F

*
  - 눈 앞에 한명의 사람이 서 있습니다. 그런데 이 사람은 커튼에 가려져 있으며 우리가 볼 수 있는 것은 커튼에 비친 그 사람의 형상뿐입니다.
  - 이 사람이 누구일까요? 철수일까요 아닐까요? 아니면 철수일까요 영희일까요 아니면 설까치일까요? 아니면 남자일까요 여자일까요?
  
  - 베이지안 확률이 사용되는 전형적인 예가 됨
  
 ML (Maximum Likelihood) : 가능한 사람들에 대해서 P(눈앞의사람|철수), P(눈앞의사람|영희), ... 등을 각각 계산해서 
 그중 likelihood가 가장 높은 사람을 선택
 MAP (Maximum A Posteriori) : posterior 확률 P(철수|눈앞의사람), P(영희|눈앞의사람), ... 을 계산해서 가장 확률이 높은 사람을 선택
 
 
**나이브 베이즈 예제1**


  - 우선 텍스트로 벡터를 구현해보자.
  
```python
from numpy import *

def loadDataSet() :
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] # 1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet) :
    vocabSet = set([]) # create empty set
                        # 이 변수에 각 문서로부터 새로운 집합 유형의 변수를 생성하여 첨부
    for document in dataSet :
        vocabSet = vocabSet | set(document) #union the two sets
    return list(vocabSet)
    
def setOfWords2Vec(vocabList, inputSet) :
    returnVec = [0] * len(vocabList)
    for word in inputSet :
        if word in vocabList :
            returnVec[vocabList.index(word)] = 1
        else : print("The word : {} is not in my Vocabulary".format(word))
    return returnVec
```

```python
list0Posts, listClasses = loadDataSet()
list0Posts
```

  - 이렇게 찍으면 list 에 vocab 들이 쭉 저장되고,
  - [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
  
     ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
     
     ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
     
     ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
     
     ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
     
     ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
     
```python
myVocabList = createVocabList(list0Posts)
myVocabList
```

  - 이렇게 찍으면 하나의 리스트에 vocab 들이 쭉 저장된다.
  - ['him',
 'ate',
 'cute',
 'food',
 'dog',
 'park',
 'posting',
 'licks',
 'quit',
 'to',
 'I',
 'so',
 'my',
 'garbage',
 'help',
 'not',
 'problems',
 'worthless',
 'mr',
 'steak',
 'is',
 'stupid',
 'take',
 'love',
 'stop',
 'flea',
 'how',
 'has',
 'dalmation',
 'please',
 'maybe',
 'buying']
 
```python
setOfWords2Vec(myVocabList, list0Posts[0])
```

  - 이걸 찍으면 0 아니면 1 값들이 하나의 리스트 안에 저장됨 (is abusive or not 의 값들)
  
  - 단어 벡터들이 생성이 되었으면, 이거로 확률을 계산해야 하는데
  
  
*단어 벡터들로 확률을 계산하기 위해서는...*


  - Count the number of documents in each class

  // 각 분류 항목에 대한 문서의 항목세기

  for every training document :

  // 훈련을 위한 모든 문서의 개수만큼 반복

      for each class :

      // 분류 항목 개수만큼 반복

          if a token appears in the document -> increment the count for that token

          // 해당 토큰이 문서 내에 있으면 -> 해당 토큰에 대한 갯수 증가

          increment the count for tokens

         //토큰에 대한 개수 증가

     for each class :

      // 분류 항목 개수만큼 반복

         for each token :

          // 토큰 개수만큼 반복

              divide the token count by the total count to get conditional probabilities

              // 조건부 확률을 구하기 위해 해당 토큰의 개수를 토큰 전체의 개수로 나눔

      return conditional probabilities for each class

      // 각 분류 항목에 대한 조건부 확률을 반환
  
  
    - 위의 알고리즘을 토대로 단어 벡터로 확률을 계산해보기.
    
    
```python
# 단어 벡터로 확률 계산하기

def trainNB0(trainMatrix, trainCategory) :
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0;
    
    for i in range(numTrainDocs) : 
        if trainCategory[i] == 1 :
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else :
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        
    p1Vect = (p1Num / p1Denom)
    p0Vect = (p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive
```

```python
trainMat=[]
for postinDoc in list0Posts :
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    
p0V, p1V, pAb = trainNB0(trainMat, listClasses)
pAb
p0V
p1V
```

  - pAb = 0.5 라는 확률이 도출됨
  - p0V : array([0.08333333, 0.04166667, 0.04166667, 0.        , 0.04166667,
  
          0.        , 0.        , 0.04166667, 0.        , 0.04166667,
       
          0.04166667, 0.04166667, 0.125     , 0.        , 0.04166667,
           
          0.        , 0.04166667, 0.        , 0.04166667, 0.04166667,
       
           0.04166667, 0.        , 0.        , 0.04166667, 0.04166667,
       
           0.04166667, 0.04166667, 0.04166667, 0.04166667, 0.04166667,
       
          0.        , 0.        ])
          
  - p1V : array([0.05263158, 0.        , 0.        , 0.05263158, 0.10526316,
  
          0.05263158, 0.05263158, 0.        , 0.05263158, 0.05263158,
          
           0.        , 0.        , 0.        , 0.05263158, 0.        ,
           
          0.05263158, 0.        , 0.10526316, 0.        , 0.        ,
          
          0.        , 0.15789474, 0.05263158, 0.        , 0.05263158,
          
          0.        , 0.        , 0.        , 0.        , 0.        ,
          
          0.05263158, 0.05263158])
          
  
  - 그래서 이게 무슨 뜻이냐면,
  - 어휘집에 있는 단어 중 'cute' 는 분류항목 0인 리스트에 한 번만 나타나고 1에는 나타나지 않음 (0.04166667)
  - 가장 큰 확률(0.15789474) 은 분류항목 1의 마지막에 나타난다. 즉 'stupid' 
  - 즉, 'stupid'는 폭력적인 분류 항목에 잘 나타난다.
  
  
**실제 조건을 반영하기 위해 분류기 수정하기**


  - 주어진 분류 항목에 속하는 문서의 확률을 구하기 위해, 많은 양의 확률들을 가지고 곱하기를 수행함
  - 이들 숫자 중 하나가 0이라면 결과는 0
  - 이러한 영향력을 줄이기 위해 발생하는 단어의 개수를 모두 1로 초기화. 분모는 2로 초기화한다.
  - trainNBO
    - p0Num = ones(numWords)
    - p1Num = ones(numWords)
    - p0Denom = 2.0
    - p1Denom = 2.0
  - p(w0|1)p(w1|1)... 처럼 매우 작은 수들을 곱할 때 underflow 가 발생하거나 부정확한 답을 산출
  - 이를 해결하기 위해, '자연로그' 를 사용한다.
    - p1Vect = log(p1Num / p1Denom)
    - p0Vect = log(p0Num / p0Denom)
    
    
```python
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1) :
    p1 = sum(vec2Classify * p1Vec) + log(pClass1) # element = wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    
    if p1 > p0 :
        return 1
    else :
        return 0
    
def testingNB() :
    list0Posts, listClasses = loadDataSet()
    testEntry = ['love', 'my', 'dalmation', 'jimain']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print("{} classified as : {}\n".format(testEntry, classifyNB(thisDoc, p0V, p1V, pAb)))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print("{} classified as : {}\n".format(testEntry, classifyNB(thisDoc, p0V, p1V, pAb)))
```

```python
testingNB()
```

  - The word : jimain is not in my Vocabulary
  
    ['love', 'my', 'dalmation', 'jimain'] classified as : 0

    ['stupid', 'garbage'] classified as : 1
    
  - 와 같이, 실제 조건에 분류기가 반영이 되는 것을 볼 수 있다.
  

   
   
   

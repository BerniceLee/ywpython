# 지도학습 기법2

### Naive Bayes

##### Bayes Theory (베이지안 이론)

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
  
  
> 그니까, 베이즈 정리는 "특성들 사이의 독립" 을 가정한다!


##### 나이브 베이즈란?
 
 
(참고 URL :  https://ko.wikipedia.org/wiki/%EB%82%98%EC%9D%B4%EB%B8%8C_%EB%B2%A0%EC%9D%B4%EC%A6%88_%EB%B6%84%EB%A5%98)

  - 나이브 베이즈는 "조건부 확률" 모델이다.
  
![조건부확률2](https://t1.daumcdn.net/cfile/tistory/999A41335A1CE91C1C)

  - 데이터 B의 관점에서 가설 A의 확률을 추정한다.
  
  - P(A) : 사전 확률, 데이터를 보기 전의 가설 확률
  - P(A|B) : 사후 확률, 데이터를 확인한 이후의 가설 확률
  - P(B|A) : 우도(조건부 확률), 데이터가 가설에 포함될 확률
  - P(B) : 한정한수(evidence), 어떤 가설에도 포함되는 데이터의 비율
  
  - 베이지안 확률의 용어를 사용하여 다음과 같이도 표현이 가능하다.
  
![naive bayesian1](https://wikimedia.org/api/rest_v1/media/math/render/svg/679e25db34f602d562e503af6d772125f78ab31e)

  - (posterior : 사후 확률, prior : 사전확률, likelihood : 우도, evidence : 관찰값)
  
  
> 모든 나이브 베이즈 분류기들은 "모든 특성 값은 서로 독립임" 을 가정한다.


  - 특정 과일을 사과로 분류 가능하게 하는 특성들 (둥글다, 빨갛다, 지름 10cm)은 나이브 베이즈 분류기에서 특성들 사이에서 발생할 수 있는 연관성이 없음을 가정하고 
  - 각각의 특성들이 특정 과일이 사과일 확률에 독립적으로 기여 하는 것으로 간주한다.
  
  
##### 나이브 베이지안 모델의 모수 추정 / 이벤트 모델


**가우시안 나이브 베이즈**


  - 연속적인 값을 지닌 데이터를 처리 할때, 전형적으로 각 클래스의 연속적인 값들이 가우스 분포를 따른다고 가정
  - 예를 들어, 트레이닝 데이터가 연속적인 속성 x를 포함하는 것으로 가정하면, 
  - 먼저 클래스에 따라 데이터를 나눈 뒤에, 각 클래스에서 x의 평균과 분산을 계산한다
  - 클래스 c와 연관된 x 값의 평균을 mu _{c}이라고 하고, 분산을 sigma _{c}^{2}라고 하면, 
  - 주어진 클래스의 값들의 확률 분포가 M과 S로 매개변수화되어 정규분포식을 통해 계산 될 수 있다
  
![Gausian naive bayes](https://wikimedia.org/api/rest_v1/media/math/render/svg/af5b6dedeaac6ad6c17fad24b6e2d172e25acde3)

  - x 벡터의 원소가 모두 실수이고 클래스마다 특정한 값 주변에서 발생한다고 할 때 주로 사용


**다항분포 나이브 베이즈**


  - 다항 이벤트 모델에서 샘플(특성 벡터)들은 다항분포 p_{1},....,p_{n} 에 의해 생성된 어떤 이벤트의 빈도수를 나타낸다
    - 특정 벡터 x는 빈도수를 나타내는 히스토그램으로 생각할 수 있다.
    
![다항분포](https://wikimedia.org/api/rest_v1/media/math/render/svg/67c08ed8a03ded9bd575f535fe9f6a30262f8ab8)

  - 다항 나이브 베이즈 분류는 로그 공간에서 표현될 때, 다음과 같은 선형 분류기가 된다.
  
![다항분포2](https://wikimedia.org/api/rest_v1/media/math/render/svg/4fb553a1d83859e2283841d0c0fca28e3698ac7a)

  - 이것은 하나의 문서에서 단어의 출현을 나타내는 이벤트를 가지고 문서 분류를 하는데 사용되는 이벤트 모델이다
   - 예를 들면, 동전세트 N 번 던졌을 때, 결과 1, ..., K로부터 어느 동전셋을 던졌는지 찾아낸다던지..
  
  
**베르누이 나이브 베이즈**


  - 다변수 베르누이 이벤트 모델에서, 특성들은 입력들을 설명하는 독립적인 부울 값(이진 변수)이다
  - 다항 모델의 특성벡터가 이벤트의 빈도수를 나타내는 반면, 이 모델은 이벤트 발생 여부를 나타내는 부울 값을 가진다
  - 만일 x_i가 어휘들 중 i번째 용어의 발생유무를 표현하는 부울일 경우, 주어진 클래스 C_{k}에 대한 문서의 우도는 다음 식으로 주어진다
  
![Bernoui naive bayes](https://wikimedia.org/api/rest_v1/media/math/render/svg/57bedaa9714484d7301c49553b1b134e47729f11)

  - 이진변수의 발생이 특성으로 사용되는 문서 분류 작업에 대하여 널리 이용된다


##### scikit-learn에 구현된 나이브 베이즈 분류기


GaussianNB, BernoulliNB, MultinomialNB 셋다 있다.

  - GaussianNB 는 연속적인 어떤 데이터에도 적용이 가능하다
  - BernoulliNB와 MultinomialNB 는 이산적인 데이터에 적용 가능
  - 특히, BernoulliNB는 이진 데이터를, MultinomialNB는 카운트 데이터 (특성이 어떤 것을 헤아린 정수 카운트로, 예를 들면 문장에 나타난 단어의 횟수) 에 적용 가능
  - 또한, BernoulliNB, MultinommialNB는 대부분 텍스트 데이터를 분류할 때 사용
  
  - BernouilliNB 분류기는 각 클래스의 특성 중 0이 아닌 것이 몇 개인지 세는 작업을 한다.
  - 코드처럼, 이진 특성을 4개 가진 데이터 포인트가 4개 있을 때
    - 클래스는 0과 1, 두개
    - 출력 y의 클래스가 0인 경우 (첫번째와 세번쨰 데이터 포인트), 
      - 첫번째 특성은 0이 두 번이고 0이 아닌 것은 한번도 없다
      - 두번쨰 특성은 0이 한번이고 1도 한번
      - 같은 방식으로 두번쨰 클래스에 해당하는 데이터 포인트에 대해서도 계산
      
```python
X = np.array([[0,1,0,1], [1,0,1,1], [0,0,0,1], [1,0,1,0]])
Y = np.array([0,1,0,1])

counts = {}
for label in np.unique(y) :
  counts[label] = X[y == label].sum(axis=0)
print("특성 카운트 : \n", counts)
```

  - X, Y array 의 값들이 특성 카운트로 출력이 된다.


##### ML ＆ MAP


(참고 URL : https://m.blog.naver.com/PostView.nhn?blogId=dgmktg&logNo=220713082304&proxyReferer=https%3A%2F%2Fwww.google.com%2F


  - 눈 앞에 한명의 사람이 서 있습니다. 그런데 이 사람은 커튼에 가려져 있으며 우리가 볼 수 있는 것은 커튼에 비친 그 사람의 형상뿐입니다.
  - 이 사람이 누구일까요? 철수일까요 아닐까요? 아니면 철수일까요 영희일까요 아니면 설까치일까요? 아니면 남자일까요 여자일까요?
  
  - 베이지안 확률이 사용되는 전형적인 예가 됨
  
 - ML (Maximum Likelihood) : 가능한 사람들에 대해서 P(눈앞의사람|철수), P(눈앞의사람|영희), ... 등을 각각 계산해서 그중 likelihood가 가장 높은 사람을 선택
 - MAP (Maximum A Posteriori) : posterior 확률 P(철수|눈앞의사람), P(영희|눈앞의사람), ... 을 계산해서 가장 확률이 높은 사람을 선택
 
 
##### 나이브 베이즈 예제1


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
  - p0V : 
          array([0.08333333, 0.04166667, 0.04166667, 0.        , 0.04166667,
  
          0.        , 0.        , 0.04166667, 0.        , 0.04166667,
       
          0.04166667, 0.04166667, 0.125     , 0.        , 0.04166667,
           
          0.        , 0.04166667, 0.        , 0.04166667, 0.04166667,
       
           0.04166667, 0.        , 0.        , 0.04166667, 0.04166667,
       
           0.04166667, 0.04166667, 0.04166667, 0.04166667, 0.04166667,
       
          0.        , 0.        ])
          
  - p1V : 
          array([0.05263158, 0.        , 0.        , 0.05263158, 0.10526316,
  
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
  

   
   
   

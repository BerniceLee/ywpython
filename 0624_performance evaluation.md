# 모형평가

### 모형평가란?

  - 고려된 서로 다른 모형들 중 어느 것이 가장 우수한 예측력을 보유하고 있는지, 선택된 무형이 '임의의 모형' 보다 우수한지 등을 비교하고 분석하는 과정
  - 이 때 다양한 평가 지표와 도식을 활용, 머신러닝 적용 목적이나 데이터 특성에 따라 적절한 성능지표를 선택해야함.
  
  - 모형평가 고려사항
  
    1. 일반화 가능성 : 같은 모집단 내의 다른 데이터에 적용하는 경우, 얼마나 안정적인 결과를 제공해주는가?
    2. 효율성 : 얼마나 적은 feature를 사용하여 모형을 구축했는가?
    3. 정확성 : 모형이 실제 문제에 적용될 수 있을 만큼 충분한 성능이 나오는가? 
    
##### Confusion Matrix

![calculation in confusion matrix](https://www.dataschool.io/content/images/2015/01/confusion_matrix2.png)

(참고 URL : https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)


  - true positives (TP): These are cases in which we predicted yes (they have the disease), and they do have the disease.
  - true negatives (TN): We predicted no, and they don't have the disease.
  - false positives (FP): We predicted yes, but they don't actually have the disease. (Also known as a "Type I error.")
  - false negatives (FN): We predicted no, but they actually do have the disease. (Also known as a "Type II error.")
  
  - Accuracy: Overall, how often is the classifier correct?
    - (TP+TN)/total = (100+50)/165 = 0.91
  - Misclassification Rate: Overall, how often is it wrong?
    - (FP+FN)/total = (10+5)/165 = 0.09
    - equivalent to 1 minus Accuracy
    - also known as "Error Rate"
  - True Positive Rate: When it's actually yes, how often does it predict yes?
    - TP/actual yes = 100/105 = 0.95
    - also known as "Sensitivity" or "Recall"
  - False Positive Rate: When it's actually no, how often does it predict yes?
    - FP/actual no = 10/60 = 0.17
  - True Negative Rate: When it's actually no, how often does it predict no?
    - TN/actual no = 50/60 = 0.83
    - equivalent to 1 minus False Positive Rate
    - also known as "Specificity"
  - Precision: When it predicts yes, how often is it correct?
    - TP/predicted yes = 100/110 = 0.91
  - Prevalence: How often does the yes condition actually occur in our sample?
    - actual yes/total = 105/165 = 0.64
    
  - 주요 기초 개념들을 참고 URL에서 볼 수 있음
  
##### F1 Score

(참고 URL : https://nittaku.tistory.com/295)

![F1 score](https://t1.daumcdn.net/cfile/tistory/99CE11505B7C0BA236)

  - Recall : 행 방향
    - input 된 class에 대해, 분류기가 어떤 class로 예측을 하는가에 대한 척도
  - Precision : 칼럼 방향
    - 예측한 값들 중에서 제대로 예측했는지에 대한 척도

> F1 score 는 이 Recall 과 Precision을 이용하여 조화평균(Harmonic mean)을 이용한 score 이다.
  
![F1 score2](https://t1.daumcdn.net/cfile/tistory/9948EA435B7C0BA333)

  - 조화평균은 [ 2 * ab / a + b ] 의 공식을 가지고있다.
  - 즉, 2개의 길이에 대해서 반대편으로 선을 내릴 때, 서로 만나는 점의 길이다.

![F1 in recall, precision](https://t1.daumcdn.net/cfile/tistory/9931F54A5B7C0BA40E)

  - Recall과 Precision 에도 위와 같이 나타낼 수 있다.
  - 즉 조화평균은 단순하게 평균을 구하는 것이 아니라,
  - /"뭔가 큰 값이 있다면 패널티를 주어서, 작은값 위주로 평균을 구한다."/


##### ROC ＆ AUC

  - ROC curve : Receiver Operating Characterestic curve
  - 보통 binary classification  이나 medical application 에서 많이 쓰는 성능 척도이다.
  
![ROC curve](https://t1.daumcdn.net/cfile/tistory/99F1DE3D5B94D8191A)

  - 위의 그래프에서 빨간색이 더 좋은 성능을 나타내는 커브다.
  - 왜 빨간색 커브가 성능이 더 좋을까?
  
![AUC score](https://t1.daumcdn.net/cfile/tistory/996BAE3B5B94D81B2F)

  - AUC : Area Under the Curve, ROC 커브의 아래 면적을 구한 것
  - AUC가 큰 것이 더 성능이 좋다.
  
> ROC 커브를 쓰는 이유가 무엇일까?

  - 클래스별 분포가 다를 때, Accuracy의 단점을 보완하면서 더 자세히 보기 위해서다.
   
![ROC curve2](https://t1.daumcdn.net/cfile/tistory/990C74445B94D81C33)

  - 위의 그림처럼, 양 끝 쪽은 확실히 정상이나 암환자로 구분이 가능
  - 하지만, 가운데 쪽은 판단이 불분명한 부분이 있을 수 밖에 없다.
  
![ROC curve3](https://t1.daumcdn.net/cfile/tistory/99B2544E5B94D81D0B)

  - 이 상황에서 최선의 판단(초록색 선을 긋는것)을 하려면, 항상 error 들을 포함할 수 밖에 없다.
  
![ROC curve4](https://t1.daumcdn.net/cfile/tistory/99C1D0435B94D81F36)

  - 위 그래프에서는 차이를 보기가 더 쉽다.
  - 초록색선을 좌우로 움직일 수록 얼마나 안좋은 결과를 초래하는지에 대한 이야기를 하는것이 결국 ROC 커브!
  
  - 겹치는 부분이 많을 수록 직선에 가까워진다.
  
![ROC curve5](https://t1.daumcdn.net/cfile/tistory/99F63A345B94D82111)


**ROC 커브에서의 수식**


![ROC curve6](https://t1.daumcdn.net/cfile/tistory/99EC523D5B94D82210)

  - 기본적으로 ROC 커브는 Recall에 대한 이야기다
  - Recall 이 높다는 것은, 초록색 선을 쭉 왼쪽으로 당겨서 실제 Positive 분포(TP)에 초록색 우측의 예측 Positive가 다 들어오도록 만드는 행위
  - FN이 줄어들면서 TP는 증가한다.
    - 따라서 TPR이 증가한다.

![ROC curve7](https://t1.daumcdn.net/cfile/tistory/992AB43F5B94D82334)

  - 그러나 그렇게 된다면 실제 Negative 분포(TN) 중 FP가 엄청나게 높아진다. 
  - 또한 실제 거출되는 TN이 엄청나게 줄어들것이다.
  
![ROC curve8](https://t1.daumcdn.net/cfile/tistory/99A40C3F5B94D8252A)

  - 즉, Recall값을 높일 수록 TPR은 증가하지만, 그만큼 TNR은 줄어드는 **Trade-Off** 가 생길 것이다.
  
  - 이런 trade-off 관계 변화에 대해 TPN, TNR의 비율을 그래프로 그린것이 결국 ROC 커브다.
  
  - 만약, TPN과 TNR의 비율을 그린다면 아래와 같은 모양으로 나타나고
  
![ROC curve9](https://t1.daumcdn.net/cfile/tistory/99479B505B94D82601)

  - TPN과 1-TNR의 비율을 그리면, 처음 봤던 ROC 커브 그래프가 나타나게 된다.
  
![ROC curve10](https://t1.daumcdn.net/cfile/tistory/992383455B94D82815)

  - 극단적으로 판단선을 왼쪽으로 옮겨보자.
  
![ROC curve11](https://t1.daumcdn.net/cfile/tistory/99136D3E5B94D82A0A)

  - 이 경우, TN가 모두 FP가 되면서 TN = 0이 됨. 그럼 1-TNR은 1이다.
  - FN도 0이 되어서 TPR은 1이 된다.
  - 위에 ROC 커브 그래프 상에서는 우측 상단이 된다. (a) 부분
  
![ROC curve12](https://t1.daumcdn.net/cfile/tistory/996C4B3A5B94D82B35)

  - 반면에, 판단선을 가장 우측으로 밀어넣으면,
  - TP가 모두 FN이 되면서 TP = 0, TPR = 0
  - FP가 모두 TN이 됨. TNR = 1, 1-TNR = 0이 되면, 그래프 상에서는 아래와 같은 점이 찍힌다.
  
![ROC curve13](https://t1.daumcdn.net/cfile/tistory/9963E5435B94D82D02)


**ROC 커브에 대해 요약하자면**


  1. Accuracy가 다가 아니다. 클래스가 imbalanced 한 데이터면 Precision, Recall 도 봐야한다.
    - Recall 같은 경우, 1차 검사에서 예측값을 암환자로 다 가져와야 하는 경우에 잘 봐야한다.
      - 실제값에 대해 판정도 다 들어와야 하는 경우
    - Precision 의 경우, 판단(예측)이 실수하면 안되는 경우에 잘 뵈야한다.
      - 유죄판결인 경우, 모두 실제값이여야 한다.
    
  2. ROC curve는 판단 선(cut off, decision boundary)을 움직였을 때, 얼마나 성능(결정)들이 민감한지에 대한 것이다.
    - AUC가 넓을 수록, 안정된 예측을 할 수 있는 모델/데이터 이다.
   
   
##### ROC 커브 그려보기 예제

```python
# 다음과 같은 모듈들을 미리 불러주고...

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ROC 커브, AUC 스코어 불러오기
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
```

```python
# ROC 커브 그리는 함수

def plot_roc_curve(fpr, tpr) :
    plt.plot(fpr, tpr, color='orange',  label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
```

```python
data_X, class_label = make_classification(n_samples=1000, n_classes=2,
                                         weights=[1, 1], random_state=1)
trainX, testX, trainY, testY = train_test_split(data_X, class_label,
                                                test_size=0.3, random_state=1)
model = KNeighborsClassifier()
model.fit(trainX, trainY)
```

  - KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
  
           metric_params=None, n_jobs=None, n_neighbors=5, p=2,
           
           weights='uniform')
           
  - 훈련용/테스트용 X,Y 세트를 우선 분류한다. (kNN : k 최측근 분류알고리즘에 의해서 분류한다.)
  - 테스트용 모델을 완성하면, 테스트 모델을 통해 예측 ROC 커브를 그려보자.
  
```python
probs = model.predict_proba(testX)
probs = probs[:, 1]

roc = roc_curve(testY, probs)
print('ROC : {}'.format(roc))
auc = roc_auc_score(testY ,probs)
print('AUC : %.2f'% auc)
```

  - ROC : (array([0.        , 0.01408451, 0.0915493 , 0.18309859, 0.33098592,
  
       0.5       , 1.        ]), array([0.        , 0.2721519 , 0.65189873, 0.89873418, 0.98101266,
       
       0.99367089, 1.        ]), array([2. , 1. , 0.8, 0.6, 0.4, 0.2, 0. ]))
       
    AUC : 0.91
    
  - testX set 을 기준으로, ROC / AUC 값들을 구할 수 있다.
  - 이것들을 시각화해주자.
  
```python
fpr, tpr, thresholds = roc_curve(testY, probs)
plot_roc_curve(fpr, tpr)
```

  - 실행결과는 ipynb 에서
  
  
##### k-겹 교차검증 (k-fold cross-validation)


  - k는 특정 숫자로 보통 5 또는 10을 사용한다.
  - 데이터를 fold 라고 하는 비슷한 크기의 '부분 집합' 으로 나눈다.
  
  
```python
# k-fold cross-validation 을 위한 사전 모듈추가

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.linear_model import LogisticRegression
```

```python
iris = load_iris()
KNN = KNeighborsClassifier()
scores = cross_val_score(KNN, iris.data, iris.target)
print("교차 검증 점수 : ", scores)
```

  - 교차 검증 점수 :  [0.98039216 0.98039216 1.        ]
  
```python
# logreg = LogisticRegression()

# scores = cross_val_score(logreg, iris.data, iris.target, cv = 5)
scores = cross_val_score(KNN, iris.data, iris.target, cv = 5)
print("교차 검증 점수 : ", scores)
```

  - 교차 검증 점수 :  [0.96666667 1.         0.93333333 0.96666667 1.        ]
  
```python
print("교차 검증 평균 점수 : {:.2f}".format(scores.mean()))
```

  - 교차 검증 평균 점수 : 0.97

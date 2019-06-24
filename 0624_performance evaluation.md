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

(참고 URL2 : https://bcho.tistory.com/1206)


  - true positives (TP): These are cases in which we predicted yes (they have the disease), and they do have the disease.
  - true negatives (TN): We predicted no, and they don't have the disease.
  - false positives (FP): We predicted yes, but they don't actually have the disease. 
    - (Also known as a "Type I error.")
  - false negatives (FN): We predicted no, but they actually do have the disease. 
    - (Also known as a "Type II error.")
  
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


  - K개의 fold 를 만들어서 진행하는 교차검증
  
> 왜 k-fold 교차검증을 사용할까?

  1. 총 데이터 갯수가 적은 데이터셋에 대해 정확도를 향상시킬 수 있음
  2. 이는 기존에 Training / Validation / Test 세 개의 집단으로 분류하는 것 보다, Training과 Test 로만 분류할때 데이터 셋이 더 많기 때문이다.
  3. 데이터 수가 적은데 검증과 테스트에 데이터를 더 뺏겨버리게 되면, underfitting 등 성능이 미달되는 모델이 학습이 되기 때문이다.
  
  - 그림을 통해 살펴보자
  
![k-fold validation1](https://blogfiles.pstatic.net/MjAxODA2MTNfMTkg/MDAxNTI4ODY1MzQzMTAw.mm2vnmN4VV_6-v1W84YkqBlwxcnTdS73f7ZN0gDkDOsg._7jgav1jx55qa7BrrgFkTgqWfrioQn5kjgejCA3rSx8g.PNG.ssdyka/ch05_%EA%B5%90%EC%B0%A8%EA%B2%80%EC%A6%9D.png?type=w2)

  - 5겹 교차검증일 때 검증 과정은 각 fold 마다 한 단계씩을 거치게 된다.
  
> k-fold 교차검증을 하는 과정

  1. 기존 과정과 같이 Training Set과 Test Set을 나눈다.
  2. Training을 K개의 fold 로 나눈다.
  3. 위는 5개의 fold로 나눴을 때 모습
  4. 한 개의 fold에 있는 데이터를 다시 K개로 쪼갠 다음, K-1개는 Training Data, 마지막 한개는 Validation Data Set 으로 지정한다.
  5. 모델을 생성하고 예측을 진행, 이에 대한 에러값을 추출한다.
  6. 다음 fold 에서는 Validation set을 바꿔서 지정, 이전 Fold 에서 Validation 역할을 했던 Set은 다시 Training Set 으로
  7. 이를 K번 반복한다.
  
  (기존 kNN 파트 부분 설명을 참조하자)
  
  8. 각각의 fold의 시도에서 기록된 Error 을 바탕(에러들의 평균)으로 최적의 모델(조건)을 찾는다
  9. 해당 모델(조건)을 바탕으로 전체 Training set의 학습을 진행한다.
  10. 해당 모델을 처음에 분할하였던 Test set을 활용하여 평가한다.
  
> k-fold 교차검증 에서는...

  - k는 특정 숫자로 보통 5 또는 10을 사용한다.
  - 데이터를 fold 라고 하는 비슷한 크기의 '부분 집합' 으로 나눈다.
  
  - 예시를 통해서 살펴보자.
  
  
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


### Exploratory Data Analysis


**타이타닉 데이터를 이용한 분석, 시각화 실습**


  - 실제 타이타닉의 탑승객의 데이터를 가지고, 분석 모델 예측을 해보자.
  - 타이타닉의 승객들의 정보들을 담아볼 수 있고, 이를 시각화 해서 나타내볼 수 있음
  - 실행 전체 결과는 ipynb 통해서 볼 수 있음
  
  
**안전 운전자 예측 경진대회 데이터를 이용한 분석 실습**

  - 훈련 데이터에는 59만명의 운전자에 관련한 데이터가 포함되어있다.
  - 테스트 데이터에는 89만명 가량의 운전과 관련된 데이터가 포함
  - 테스트 데이터에는 운전자의 보험 청구 여부를 나타내는 'target'  변수를 포함하고 있지 않아서,
    - 훈련 데이터보다 변수가 하나 적은 58개이다.
    
```python
import pandas as pd
import numpy as np

trn = pd.read_csv('C:/Users/Affinity/Desktop/개인자료/module3/ssd_train.csv', 
                  na_values=['-1', '-1.0'])
tst = pd.read_csv('C:/Users/Affinity/Desktop/개인자료/module3/ssd_test.csv', 
                  na_values=['-1', '-1.0'])

# 데이터 크기 확인
print(trn.shape, tst.shape)
```

  - (595212, 59) (892816, 58)
  - 실제 데이터 셋의 차이 갯수가 하나씩 나는걸 알 수 있다. (왼쪽은 size)
  
```python
# 데이터 첫 5줄 확인
trn.head()

# 데이터프레임에 대한 메타 정보를 확인한다
trn.info()
```

  - 
    <class 'pandas.core.frame.DataFrame'>
  
    RangeIndex: 595212 entries, 0 to 595211
  
    Data columns (total 59 columns):
  
    id                595212 non-null int64
   
    target            595212 non-null int64
   
    ps_ind_01         595212 non-null int64
   
    ps_ind_02_cat     594996 non-null float64
  
    ps_ind_03         595212 non-null int64
  
    ps_ind_04_cat     595129 non-null float64
  
    ps_ind_05_cat     589403 non-null float64
  
  - 이런식의 결과가 나온다.
  
  - 대부분의 변수가 수치형이다.
  - 변수명이 'ps_ind_..' 형태로 익명화 되어있음
  - 경진대회 주최 측에서 고객의 개인정보 보호를 위하여 철저하게 익명화 한 것으로 파악
  
  - 익명화된 변수명을 통해 변수의 형태를 짐작할 수 있음
    - bin 으로 끝나는 변수는 이진(binary) 변수이고,
    - cat 으로 끝나는 변수는 범주형(category) 변수
    - -1 값은 결측값, 데이터를 불러오는 과정에서 NaN으로 지정
    
    
```python
np.unique(trn['target'])
# 실행 결과 : array([0, 1], dtype=int64)

1.0 * sum(trn['target']) / trn.shape[0]
# 실행 결과 : 0.036447517859182946
```

  - 타겟 변수의 고유값은 보험 청구 여부를 나타내는 [0,1]중 하나의 값을 가지는 이진 변수
  - 전체 데이터 중 3.6%의 운전자가 보험 청구를 진행함
  - 문제 특성상, 타겟 변수가 1일 확률이 매우 낮은, 불균형한 데이터
  
```python
# 훈련 데이터와 테스트 데이터를 통합
tst['target'] = np.nan
df = pd.concat([trn, tst], axis=0)
```

```python
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# 시각화 함수 미리 정의
def bar_plot(col, data, hue=None) :
    f, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x=col, hue=hue, data=data, alpha=0.5)
    plt.show()
    
def dist_plot(col, data) :
    f, ax = plt.subplots(figsize=(10, 5))
    sns.distplot(data[col].dropna(), kde=False, bins=10)
    plt.show()
    
def bar_plot_ci(col, data) :
    f, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=col, y='target', data=data)
    plt.show()
```

```python
# 분석의 편의를 위해 변수 유형별로 구분한다.

# 이진변수 작성
binary = []
for index in range(6, 19) :
    binary.append("ps_ind_{}_bin".format(index))
for index in range(14, 16) :
    binary.remove("ps_ind_{}_bin".format(index))        
for index in range(15, 21) :
    binary.append("ps_calc_{}_bin".format(index))
    
# 범주형변수 작성
category = []
for index in range(2, 6) :
    category.append("ps_ind_{}_cat".format(index))
for index in range(3, 4) :
    category.remove("ps_ind_{}_cat".format(index))        
for index in range(1, 12) :
    category.append("ps_car_{}_cat".format(index))

# 정수형 변수
integer = []
for index in range(1, 4) :
    integer.append("ps_ind_{}".format(index))
for index in range(2, 3) :
    integer.remove("ps_ind_{}".format(index))
for index in range(14, 16) :
    integer.append("ps_ind_{}".format(index))
for index in range(4, 15) :
    integer.append("ps_calc_{}".format(index))
integer.append("ps_car_{}".format(11))    

# 소수형 변수
floats = []
for index in range(1, 4) :
    floats.append("ps_reg_{}".format(index))
for index in range(1, 4) :
    floats.append("ps_calc_{}".format(index))        
for index in range(12, 16) :
    floats.append("ps_car_{}".format(index))
```

```python
print(binary,"\n\n", category,"\n\n", integer,"\n\n", floats)
```

  - binary 는 이러한 값이 담긴다.
     ['ps_ind_6_bin', 'ps_ind_7_bin', 'ps_ind_8_bin', 'ps_ind_9_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 
 
    'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_calc_15_bin', 
  
    'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin'] 

  - category 는 이러한 값이 담긴다.
     ['ps_ind_2_cat', 'ps_ind_4_cat', 'ps_ind_5_cat', 'ps_car_1_cat', 'ps_car_2_cat', 'ps_car_3_cat', 
    
    'ps_car_4_cat', 'ps_car_5_cat', 'ps_car_6_cat', 'ps_car_7_cat', 'ps_car_8_cat', 'ps_car_9_cat', 
   
    'ps_car_10_cat', 'ps_car_11_cat'] 

  - integer 는 이러한 값이 담긴다.
    ['ps_ind_1', 'ps_ind_3', 'ps_ind_14', 'ps_ind_15', 'ps_calc_4', 'ps_calc_5', 'ps_calc_6', 'ps_calc_7', 
    
    'ps_calc_8', 'ps_calc_9', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 
   
    'ps_car_11'] 

  - floats 는 이러한 값이 담긴다.
    ['ps_reg_1', 'ps_reg_2', 'ps_reg_3', 'ps_calc_1', 'ps_calc_2', 'ps_calc_3', 'ps_car_12', 'ps_car_13', 
    
    'ps_car_14', 'ps_car_15']
    
    
```python
# 시각화

for col in binary + category + integer :
    bar_plot(col, df)
```

  - 출력 결과는 ipynb 에서

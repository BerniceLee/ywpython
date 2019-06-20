### Supervised vs unsupervised
  
  
![supervised vs unsupervied](http://solarisailab.com/wp-content/uploads/2017/06/supervsied_unsupervised_reinforcement.jpg)
  
  
	
##### 지도학습 vs 비지도학습, 어떤 차이가 있는가?



> 모형을 통해 target value를 예측하려고 하는가, 아닌가? 에 따라 갈린다.

	- target value 를 예측하려고 한다면 -> supervised
		- target value 가 discrete(이산) -> 분류
		- target value 가 continuous(연속) -> 회귀
			
			* 사실상 분류/회귀는 개념적인 분류일 뿐, 같은 개념이다.
			
	- target value 를 통해 예측하지 않는다 -> unsupervised
		- 어떤 discrete group 에 알맞은지를 보려면? -> 군집화
		- 각각의 group에 알맞은 정도가 어느 정도인지 수치적으로 평가하고자 한다 -> 밀도추정
	
	
	
##### 그렇다면 지도학습 vs 비지도학습, 어떤 기법을 사용할 것인가?



	- 앞서 말한 target value를 예측하려는건지 여부도 중요하지만,
	- 재료가 되는 데이터의 특성을 살펴보는 것도 중요하다.
		- 속성값의 타입, 변수 종류, NA량, NA핸들링의 논리, outlier 패턴, 바빈발항목의 패턴 등...
		
	- 또한 결과 해석의 목적/실용성에 따라 기법을 선택하기도 한다.
		- 예를 들어, 새 점포의 입지선정을 위한 분석이면,
			- 영수증 거래데이터, 인구센서스, 유동인구, 공간데이터를 기반으로,
			- 트리계열로 prediction 하는 것이 타당
			- but, 새로운 설명변수로 사용하기 위해 매출데이터를 물품, 시간대별 판매량 등으로 clustering 해볼 수도 있음



##### 각각의 예시



**Supervised : classification and regression (분류와 회귀)**


	- 스팸메일 거르기
	
![spam mail1](https://tensorflowkorea.files.wordpress.com/2018/05/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2018-05-23-e1848be185a9e18492e185ae-11-57-23.png?w=768)
![spam mail2](https://tensorflowkorea.files.wordpress.com/2018/05/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2018-05-24-e1848be185a9e1848ce185a5e186ab-12-01-09.png?w=768)


	* 스팸 필터의 경우, 많은 메일 샘플과 소속 정보(스팸인지 아닌지)로 훈련되어야 한다.
	* 회귀 아라고리즘의 경우도 살펴보면 일정한 스팸메일에 대한 회귀선이 그려진다.


**Unsupervised : clustering and dimension reduction**


![clustering traning set](https://tensorflowkorea.files.wordpress.com/2018/05/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2018-05-24-e1848be185a9e1848ce185a5e186ab-12-03-43.png?w=768)

	* 군집 clustering
		* k-평균 k-Means
		* 계층 군집 분석 Hierarchical Cluster Analysis (HCA)
		* 기댓값 최대화 Expectation Maximization
	* 시각화visualization와 차원 축소dimensionality reduction
		* 주성분 분석Principal Component Analysis (PCA)
		* 커널kernel PCA
		* 지역적 선형 임베딩Locally-Linear Embedding (LLE)
		* t-SNEt-distributed Stochastic Neighbor Embedding
	* 연관 규칙 학습Association rule learning
		* 어프라이어리Apriori
		* 이클렛Eclat
		
		
**Semi-supervised learning**


![semi-supervised learning](https://tensorflowkorea.files.wordpress.com/2018/05/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2018-05-24-e1848be185a9e1848ce185a5e186ab-12-19-21.png?w=768)

	- 준지도 학습 : 일부는 시즈모드, 일부는 퉁퉁퉁
	
	
**Reinforced learning**


![reinforced learning](https://tensorflowkorea.files.wordpress.com/2018/05/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2018-05-24-e1848be185a9e1848ce185a5e186ab-12-21-44.png?w=768)

	- 강화 학습 : 보상과 벌점 시스템 도입
	- 학습한 기계는 시간이 지나면서 가장 큰 보상을 얻기 위해 "정책(Policy)" 라는 최상의 전략을 스스로 학습함
	
	
	
### 머신러닝의 이슈들



##### 1. Quantitiy of Training Data

	- 많은 양의 예제가 필요하다...

##### 2. Non-representative Training Data

	- 대표성의 부족
	- Garbage in, Garbage out (GiGo : 무가치한 데이터를 넣으면 무가치한 데이터가 나온다)
	
##### 3. Sampling

	- 흔히 통계에서의 샘플링은 inference 가 목적이면, 머신러닝에서는 train data의 다양성 확보가 목적
	
##### 4. Irrelevant Features

	- 데이터에 관련성이 없는 자료들 or 인스턴스가 많은 경우 학습이 제대로 되지 않는다
	
##### 5. Overfiting (to the training data)

	- 과대 적합 : 데이터에 대해 모델의 학습을 과도하게 한 경우
	
##### 6. Underfiting (to the training data)

	- 과소 적합 : 데이터에 대해 모델이 너무 단순한 경우
	
![overfiting vs underfiting](https://cdn-images-1.medium.com/max/1600/1*JZbxrdzabrT33Yl-LrmShw.png)

> Underfiting issue 에 대한 해결책?

	- 더욱 복잡하고 강한 모델을 사용하여 데이터를 학습
	- Feature Engineering을 통해, 더 좋은 Feature들을 생성하여 학습
	- 모델의 Constraint를 완화한다.
	
> Overfit, Underfit 여부는 어떻게 알 수 있을까?

![when is a model overfit/underfit?](https://i.stack.imgur.com/t0zit.png)

	- overfit 과 underfit 사이의 적정한 차수를 찾아야한다.
		- 어떻게보면, overfit 과 underfit 도 약간 trade-off 관계라서
		- 어떤 기준에 치우치면 기준선을 중심으로 치우치면 overfit, 부족하면 underfit
		
![trade-off of fitting issue](https://image.jimcdn.com/app/cms/image/transf/dimension=1920x400:format=png/path/s8ff3310143614e07/image/i21af0dd4f2772075/version/1550368786/image.png)0619 confusion matrix practice
![low bias vs high bias](https://prateekvjoshi.files.wordpress.com/2015/10/3-bulls-eye.png?w=338&h=258&zoom=2)

	* 그림을 보면, low variance + low bias 가 우리의 최종 목적이지만, 도달하기가 쉽지 않다는 것을 알 수 있음
	* 중심을 잘 찾아야..ㅎㅎ
	
> Error(X) = Noise(X) + Bias(X) + Variance(X)


##### Statistics vs. Machine Learning


	- 통계 : 설명과 이해를 위해 데이터를 활용
	- 머신러닝 : 예측을 위해서, 예측력과 설명력에 초점을 맞춤
	
![statistics vs machine	learning](https://statkclee.github.io/ml/fig/ml-basic-stat-vs-ml.png)



### 머신러닝에 필요한 수학 기초



##### 벡터

	- 숫자를 나란히 나타낸 것
	- 세로벡터 / 가로벡터
	
```python
# 1차원 벡터
import numpy as np
a = np.array([2,1])
print(a)
type(a)
```

```python
# 2차원 벡터
c = np.array([[1,2],[3,4]])
print(c)
d = np.array([[1],[2]])
print(d)
print(d.T)
```

**벡터의 내적(dot product) / 외적(cross product)**


(참조 URL : https://wikidocs.net/22384)


1. 내적, 내적의 연산

	- a⃗ ⋅b⃗ =|a⃗ ||b⃗ |cosθ
	- 벡터에는 방향이 있으므로, 방향이 일치하는 만큼만 곱한다
	- 실습코드 부분 참조
	
```python
# 벡터의 내적
b = np.array([1,3])
c = np.array([4,2])
print(b.dot(c))
```


##### 미분

	- 함수의 기울기를 구하는 과정
	- 머신러닝에서는 중첩 함수이 미분이 많다.
		- df/dg, dg/dw 등, 여러 함수에 대한 중첩 미분이 작용한다.
		
**편미분**

(참고 URL : https://datascienceschool.net/view-notebook/5f5f4a966f9042efa43d711008df9c0b/)


	- 머신러닝에서 실제로 쓰는것은 편미분
	- 복수 변수를 가지는 함수 두개가 있다고 할 때, 한 함수에만 주목하여 다른 함수를 변수로 간주하여 미분하는 것

![partial difference](https://datascienceschool.net/upfiles/816e894c32d24a458872a18b92e384c4.png)

	- 만약 함수가 두 개 이상의 독립변수를 가지는 다변수 함수인 경우에도 미분 즉, 기울기는 하나의 변수에 대해서만 구할 수 있다. 		- 따라서 편미분의 결과로 하나의 함수에 대해 여러 개의 도함수가 나올 수 있다.
	- 편미분의 결과 즉 도함수는 독립 변수를 함수의 아랫첨자로 써서 표기하거나  ∂ (round) 기호를 사용하여 표기한다.  
	- x,y  두 개의 독립 변수를 가지는 함수  f(x,y) 의 편미분 도함수는 다음과 같이 표기한다.


	- fx(x,y)=∂f∂x(4.2.18)
	- fy(x,y)=∂f∂y(4.2.19)
	
	
	- 예제 코드를 통해서 한번 알아보자
	
	
```python
# 편미분
# f를 동고선으로 표시하고, w의 공간을 격자형으로 나눴을 때 각 점에서의 경사를 화살표로 그림

import matplotlib.pyplot as plt

def f (w0, w1) :# (A) f의 정의
    return w0**2 + 2 * 2 * w0 *w1 + 3
def df_dw0(w0, w1) : # (B) f의 w0에 대한 편미분
    return 2* w0 + 2* w1
def df_dw1(wo, w1) : # (C) f의 w1에 대한 편미분
    return 2 * w0 + 0 * w1

w_range = 2
dw = 0.25

w0 = np.arange(-w_range, w_range + dw, dw)
w1 = np.arange(-w_range, w_range + dw, dw)
wn = w0.shape[0]

ww0, ww1 = np.meshgrid(w0, w1) # (D)
ff = np.zeros((len(w0), len(w1)))
dff_dw0 = np.zeros((len(w0), len(w1)))
dff_dw1 = np.zeros((len(w0), len(w1)))

for i0 in range(wn) : # (E)
    for i1 in range(wn) :
        ff[i1, i0] = f(w0[i0], w1[i1])
        dff_dw0[i1, i0] = df_dw0(w0[i0], w1[i1])
        dff_dw1[i1, i0] = df_dw0(w0[i0], w1[i1])
        
plt.figure(figsize=(9,4))
plt.subplots_adjust(wspace=0.3)
plt.subplot(1,2,1)

cont = plt.contour(ww0, ww1, ff, 10, colors='k') # (F) f의 등고선 표시
cont.clabel(fmt='%2.0f', fontsize=0)

plt.xticks(range(-w_range, w_range+1, 1))
plt.yticks(range(-w_range, w_range+1, 1))
plt.xlim(-w_range - 0.5, w_range + .5)
plt.ylim(-w_range - .5, w_range + .5)
plt.xlabel('$w_0$', fontsize=14)
plt.ylabel('$w_1$', fontsize=14)
plt.subplot(1,2,2)

plt.quiver(ww0, ww1, dff_dw0, dff_dw1) # (G) f의 경사 벡터 표시
plt.xlabel('$w_0$', fontsize=14)
plt.ylabel('$w_1$', fontsize=14)
plt.xticks(range(-w_range, w_range+1, 1))
plt.yticks(range(-w_range, w_range+1, 1))
plt.xlim(-w_range - 0.5, w_range + .5)
plt.ylim(-w_range - .5, w_range + .5)

plt.show()
```

	- 실행결과는 ipynb 파일에서 열람가능
	- 왼쪽의 f의 등고선 플롯의 수치를 보면, f는 오른쪽 위와 왼쪽 아래가 높고, 왼쪽 위와 오른쪽 아래가 낮은 지형을 볼 수 있음
	- 오른쪽이 이 지형의 경사, 화살표는 각 점에서 경사가 높은 쪽을 향함
		- 화살표의 크기로 경사의 가파름 정도를 알 수 있음
	- 경사는 그 함수의 최대점과 최소점을 찾는 데 중요한 개념
	- 머신러닝에서는 오차 함수의 최소점을 구하기 위해 오차 함수의 경사를 계산한다.
		- 후에 이 경시를 이용하니 참조


**편미분 : 다변수의 중첩 함수의 미분**


	- 보통 이 문제는 여러 층으로 된 신경망 학습 규칙을 도출할 때 쓰인다.
	
	- dzdx=∂z∂y1dy1dx+∂z∂y2dy2dx+⋯+∂z∂yNdyNdx (4.2.20)
		- x값의 변화에 따른 z값의 변화를 각 index 마다 곱해주고 합계 계산
		


##### 행렬

(참고 URL : https://namu.wiki/w/%ED%96%89%EB%A0%AC)


1. 행렬의 스칼라 연산, 상수배
	
	- 대응하는 모든 원소끼리 연산하고, 상수도 대응하는 원소에 곱해준다.
	- 행렬곱과 내적(dot product)은 다르다!
		- 벡터와 행렬은 다르다.
		- 만약에 벡터를 행렬로 변환하려면 Column vector 로 변환할때 전치가 한번 따라붙는다
	
2. 단위행렬

	- 정방 행렬, 대각선 성분이 1, 나머지는 0
	- 단위 행렬은 스칼라의 1과 비슷하다
		- 어떤 행렬(크기가 같은 정방행렬)에 단위행렬 E를 곱해도 결과가 바뀌지 않음.
		
```python
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
I = np.identity(3) # 3x3인 단위행렬
print(A.dot(I))
```

3. 행렬과 사상

	- 벡터의 덧셈과 뺄셈에 도형적인 해석이 있듯이, 행렬에도 도형적인 해석이 존재
	- 행렬은 '벡터를 다른 벡터로 변환하는 규칙'으로 해석할 수 있음
	- 벡터라는 숫자의 조합을 '좌표' 라고 해석하면, 그것은 '공간에 있는 점'으로도 볼 수 있기 때문에,
	- 행렬은 "어떤 점을 다른 점으로 이동시키는 규칙" 으로 파악이 가능하다.
	- 이처럼 그룹(벡터와 점)에서 그룹(벡터와 점)에 대응 관계를 제공하는 규칙을 사상이라고 하며, 행렬은 선형 사상으로 분류
	
![linear map](https://t1.daumcdn.net/cfile/tistory/2113A43C56ED62C101)



# 비지도학습 기법

### Association Rules



##### Association Rules (연관규칙) 이란?


다음과 같은 테이블이 있다고 가정했을 때, 연관규칙의 주요 개념은 이렇다.

| ID | Item                     |
|----|--------------------------|
| 1  | 계란, 우유               |
| 2  | 계란, 기저귀, 맥주, 사과 |
| 3  | 우유, 기저귀, 맥주, 콜라 |
| 4  | 계란, 우유, 기저귀       |
| 5  | 계란, 우유, 맥주, 콜라   |


	- 연관규칙 : '만일 맥주를 산다면, 기저귀도 산다'
		- Arule(X->Y) = 맥주 -> 기저귀	
	- 조건절 (Antecedent) : '만일 맥주를 산다면'
	- 결과절 (Consequent) : '기저귀도 산다'
	
	- 아이템 집합 = 조건절 or 결과절을 구성하는 아이템들의 집합
		- ex) 맥주 -> (계란, 기저귀, 사과)
		
	- 지지도 (Support) : 조건절이 일어날 확률
		- (X,Y를 모두 포함하는 수) / (전체 수)
		- 즉, 지지도란 '맥주를 산다면' 이 일어날 확률
			- n(2번, 3번) / N = 2/5 = 0.5

	- 신뢰도 (Confidence) : 조건절이 일어났을 때 결과절이 일어날 확률
		- (X,Y를 모두 포함하는 수) / (X를 포함한 수)
		- 즉, 신뢰도란 '맥주를 산다면 기저귀도 살 확률'
			- n(2번,3번) / n(2번, 3번, 5번) = 2/3 = 0.667
			
	- 향상도 (Lift) : 조건절과 결과절이 서로 독립일 때에 비해, 두 사건이 동시에 얼마나 발생하는가에 대한 비율
		- (신뢰도) / (Y의 신뢰도)
		- 즉, 향상도란 '우연히 기저귀를 살 확률에 비해, 맥주를 산다면 기저귀도 사는가'
		- 0.667 / 0.6 = 1.111
		
지지도를 통해 - 규칙의 빈도가 많은가, 구성비가 높은가

신뢰도를 통해 - 규칙의 조건부확률이 높은가

향상도를 통해 - 규칙이 우연에 비해 서로 관계가 있는가



**1. 일정수준 이하 지지도와 신뢰도를 가진 규칙은 제외**

**2. 향상도가 높은 순(130% 이상)으로 유용한 규칙만 추출**



> 따라서, 애초에 계산할 필요가 없는 항목집합(item set)을 줄이는 방법이 "Apriori Algorithm"

	- 선험적인 알고리즘이다 (경험하기 전 미리 작업)



##### Association Rules의 원리



![Association Rules](https://akshanshweb.files.wordpress.com/2018/04/screenshot-from-2018-04-19-12-36-54.png?resize=614%2C446)

	- 각 줄별로 하나의 항목집합(item set)
	- 각 항목집합마다 일정 Support 값,
	
일정 Support 값(보통 0.05)을 기준으로, 그 이하 값을 가진 집합은 **"가지치기(pruning)"**

![Association Rules2](https://akshanshweb.files.wordpress.com/2018/04/screenshot-from-2018-04-19-13-03-33.png?resize=594%2C367)


1차 연산 : 지지도에 따라 가지치기

	- support 값이 낮은 것들을 가지치기 하고, 남은 항목들을 각각 다른 항목집합으로 묶어준다.
	
	- 해당 항목집합만 연산에 고려하면 됨.

2차 연산 : 신뢰도에 따라 가지치기

	- 1차 연산 자료들에서 빈번한 정도에 따라 항목집합을 재조합한다.
	
	- confidence 가 미달하는 경우 역시 가지치기
	
	
1,2차 연산 이후 향상도(Lift) 리스트를 작성하고, 이 전체 과정을 반복하고,

최종적으로, Lift 값이 높은 추천리스트 (top N List) 를 생성한다.
	
	
	
	
### CF (Collaborative Filtering)



##### CF란?

추천 시스템에서 주로 사용하는 알고리즘
	

	- 고객의 상품 구매이력
	- 함께 구매한 규칙 (상품A -> 동시에 상품B)를 찾음
	- 고객이 특정 상품 구매 시 이와 연관성 높은 상품을 추천
	
	- 고객의 상품 구매 이력
	- 고객 A와 다른 고객B의 상관계수를 비교
	- 서로 높은 상관이 인정되는 경우 고객B가 구입한 상품 중에 고객A가 미구입한 상품을 고객A에게 추천
	
	
![CF](https://t1.daumcdn.net/cfile/tistory/255A8E4E5350CE4B27)

	- 아이템에 대한 선호도를 계산 = 유사성
	- 유사성의 계산은 상품 구입 빈도, 유저의 상품 평가, 상품 클릭 횟수 등
	
CF에는 User Based, Item Based 두 가지가 있다.

![User based](https://t1.daumcdn.net/cfile/tistory/210AC8485350CE6123)

	- 나와 취향이 비슷한 유저가 무얼 구매했는지 기반으로 추천한다.
	
![Item based](https://t1.daumcdn.net/cfile/tistory/2731204B5350CE7C0D)

	- 내가 이전에 구매했던 항목을 기반으로 연관성 있는 다른 상품을 추천한다
	
Item Based CF는 Arules와 어느정도 비슷하다

하지만, Item Based CF는 전체 데이터의 item set 정보를 고려하기 어렵고 복잡해진다. 

	- 이럴때는 Arules를 사용하는게 낫다.
	 
> Arules의 장점은?

1. Item Based CF 보다는 전체 데이터를 손쉬운 계산으로 사용
2. User Based CF는 처음 시작한 유저, 전체 데이터 자체가 작을 때 작동하지 않음 (=cold start problem)
3. Arules는 전체 데이터셋의 규칙을 사용하므로 small data일때 더욱 안정적
4. Arules는 I/O를 응용, 변형, 재처리하기 쉬움. 최초 개발 단계이기 때문에 더 적절..
5. Arules는 향후 2-mode network, 자연어 처리 데이터를 모형에 반영하기 쉬움 (rule = network)
	
	
	- 이기는 한데, 그래도 연관규칙 보다는 CF를 더 많이 쓴다 ^^
	
	
	
### Clustering



##### Clustering 이란?

데이터셋을 부분으로 적절히 나누는 기법

	- 이 때 구분된 부분집합을 "군집(cluster)" 라고한다.
	- 한 데이터 포인트는 한 군집에 속함
	
![Clustering](https://www.imperva.com/blog/wp-content/uploads/sites/9/2017/07/k-means-clustering-on-spherical-data-1v2.png)


**Data-Driven**으로 유사성/거리 등을 구해서 집단을 구분한다.

	- 잘 된 구분이라면?
		- 집단 내에서는 거리가 가깝고
		- 집단 간에는 거리가 멀 것이다.
		
		
### K-means



대표적인 clustering 기법

	- K는 hyperparameter,
	- K개 그룹으로 나누기 위해 K개의 점을 찍고
	- 각각의 데이터 포인트들이 그 중심과 얼마나 가까운가를 cost 로 정의한다.
	
k-means는 이 cost 를 최소화 하는 문제다.	

![k-means](https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/ClusterAnalysis_Mouse.svg/675px-ClusterAnalysis_Mouse.svg.png)

> 과연 몇개로 군집화 해야하는가?

	- Inertia value = 군집화가 된 후에 각 중심점에서 군집의 데이터간 거리
		- 즉, 군집의 응집도를 의미
		- 이 값이 작을 수록 응집도가 높게 군집화가 잘 되었다고 평가할 수 있음
		
		- 보통 3~5 정도가 적절하다 봄
		
		
		
##### k-means 예제


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# 데이터 생성

np.random.seed(1)
N = 100
K = 3
T3 = np.zeros((N,3), dtype=np.uint8)
X = np.zeros((N,2))
X_range0 = [-3, 3]
X_range1 = [-3, 3]
X_col = ['cornflowerblue', 'black', 'white']
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]]) # 분포의 중심
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]]) # 분포의 분산
Pi = np.array([0.4, 0.8, 1]) # 누적 확률

for n in range(N) :
    wk = np.random.rand()
    
    for k in range(K) :
        if wk < Pi[k] :
            T3[n, k] = 1
            break
    for k in range(2) :
        X[n, k] = (np.random.randn() * Sig[T3[n, :] == 1, k] + Mu[T3[n, :] == 1, k])
				
# 데이터 그리기

def show_data(x) :
    plt.plot(x[:, 0], x[:, 1], linestyle='none',
             marker='o', markersize=6,
             markeredgecolor='black', color='gray', alpha=0.8)
    plt.grid(True)
    
plt.figure(1, figsize=(4, 4))
show_data(X)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.show()

# np.savez('data_ch9.npz', X=X, X_range0=X_range0, X_range1=X_range1)
```


	- 크게는 이런식이고... k-means 를 단계별로 실습 해보자.
	
1. 변수의 준비와 초기화

	- k번째 클러스터 중심 벡터 Mu = [Mux, Muy]
	- k=3으로 설정해보고, 3개의 중심 벡털ㄹ Mu0, Mu1, Mu2
	- 클래스 지시 변수 R은 각 데이터가 어느 클래스에 속해있는지 나타낸 행렬
	
	- R = 1 (데이터 n이 k에 속하는 경우) / 0 (데이터 n이 k에 속하지 않는 경우)
	 
	- 데이터 n에 대한 클래스 지시 변수를 벡터로 나타내면, 클래스 0에 속하는 경우
	
	- Rn = [Rn0, Rn1, Rn2] = [1, 0, 0]
	

```python
#  Mu 및 R 초기화

Mu = np.array([[-2,1], [-2,0], [-2,-1]])
R = np.c_[np.ones((N,1), dtype=int), np.zeros((N,2), dtype=int)]
```

	- 첫번째 문장에서 정의한 Mu는 3개의 Mu를 한 덩어리로 엮은 3x2 행렬
	- 두번째 문장에선 모든 데이터가 클래스 0에 속하도록 R을 초기화했지만
	- R은 Mu로 결정되기 때문에, 어떻게 초기화하던 다음 알고리즘에 영향을 주지 않음
	
	- 초기화 했으면 입력 데이터를 그려보자
	
	
```python
# 데이터를 그리는 함수

def show_prm(x, r, mu, col) :
    for k in range(K) :
        # 데이터 분포의 묘사
        plt.plot(x[r[:, k] == 1, 0], x[r[:, k] == 1, 1],
                marker='o',
                markerfacecolor=X_col[k], markeredgecolor='k',
                markersize=6, alpha=0.5, linestyle='none')
        # 데이터의 평균을 star mark로 표시
        plt.plot(mu[k, 0], mu[k, 1], marker='*',
                 markerfacecolor=X_col[k], markersize=15,
                 markeredgecolor='k', markeredgewidth=1)
        
    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.grid(True)
    

plt.figure(figsize=(4, 4))
R = np.c_[np.ones((N, 1)), np.zeros((N, 2))]
show_prm(X, R, Mu, X_col)
plt.title("Initial Mu and R")
plt.show()
```


이제 여기서 Mu를 R로 갱신해보자.

	- 각 별은 클러스터0, 클러스터1, 클러스터2 를 상징함
	- 데이터 X와 각 클러스터 중심 Mu 까지의 제곱 거리를 각 k에 대해 계산해준다.
		- 제곱거리 : (X와 mu의 차)**2 + (X+1과 mu+1의 차)**2
		
	- 거리가 가장 가까운 클래스를 그 데이터의 소속 클래스로 한다.
	
		- R을 갱신 : 각 데이터 점을 가장 중심이 가까운 클러스터에 넣는다.
	
우선 첫번째 데이터 점[-0.14, 0.87]을 살펴보자.
	
	- 첫번째 데이터점에서 클러스터 중심(별표) 까지의 제곱 거리를 각 클러스터에 대해 계산하고,
		- 거리 자체가 중요하다기 보다는...
		- 데이터 점에서 가장 가까운 클러스터를 알면 좋다.
	- 그 결과 클러스터 0,1,2 까지의 제곱 거리는 각각 3.47, 4.20, 6.93이다.
	- 따라서 R1 = [1,0,0]
		- 클래스 0에 속한다.
		
	- 그 다음엔 Mu를 R로 갱신한다.
	
	
```python
# Mu로 R 갱신
# R을 정한다

def step1_kmeans(x0, x1, mu) :
    N = len(x0)
    r = np.zeros((N, K))
    for n in range(N) :
        wk = np.zeros(K)
        
        for k in range(K) :
            wk[k] = (x0[n] - mu[k, 0])**2 + (x1[n] - mu[k, 1])**2
        r[n, np.argmin(wk)] = 1 
    return r

plt.figure(figsize=(4, 4))
R = step1_kmeans(X[:, 0], X[:, 1], Mu)
show_prm(X, R, Mu, X_col)
plt.title("Step 1")
plt.show()
```


그 다음엔 Mu를 갱신한다.

	- 각 클러스터에 속하는 데이터 점의 중심을 새로운 Mu로 정의한다.
	- 먼저 k=0 에 속하는 데이터, 즉 Rn = [1,0,0] 라벨을 지닌 데이터 점에 주목해서 각 "평균" 을 구한다.
	- 동일한 절차를 k=1, k=2에 대해 실시하면 된다.

	
	
```python
# Mu 의 갱신
# Mu 결정

def step2_kmeans(x0, x1, r) :
    mu = np.zeros((K, 2))
    for k in range(K) :
        mu[k, 0] = np.sum(r[:, k] * x0) / np.sum(r[:, k])
        mu[k, 1] = np.sum(r[:, k] * x1) / np.sum(r[:, k])
    return mu

plt.figure(figsize=(4, 4))
Mu = step2_kmeans(X[:, 0], X[:, 1], R)
show_prm(X, R, Mu, X_col)
plt.title("Step 2")
plt.show()
```


이 전체과정을 그려보면 다음과 같다.


```python
# 전체과정 그려보기

plt.figure(1, figsize=(10, 6.5))
Mu = np.array([[-2,1], [-2,0], [-2,-1]])
max_it = 6 # 반복 횟수
for it in range(0, max_it) :
    plt.subplot(2, 3, it+1)
    R = step1_kmeans(X[:, 0], X[:, 1], Mu)
    show_prm(X, R, Mu, X_col)
    plt.title("{0:d}".format(it + 1))
    plt.xticks(range(X_range0[0], X_range0[1]), "")
    plt.yticks(range(X_range1[0], X_range1[1]), "")
    Mu = step2_kmeans(X[:, 0], X[:, 1], R)
    
plt.show()    
```


왜곡 척도 살펴보기

**왜곡척도 (Distortion Measure)** : 데이터 점이 속한 클러스터의 중심까지의 전체 거리를 전체 데이터로 합한 것

	- 이 척도가 목적 함수에 대응한다.
	- 학습이 진행됨에 따라 감소하는 목적함수
	
![목적함수](https://wikimedia.org/api/rest_v1/media/math/render/svg/b8e266670728853e0c490c4e621ba97fc6c88f74)

	- 예시 코드에서는 목적 함수를 J로 두겠다.
	
	
```python
# 왜곡척도 - 초기값의 왜곡척도 구하기

# 목적함수

def distortion_measure(x0, x1, r, mu) :
    # 입력은 2차원으로 제한하고 있다
    N = len(x0)
    J = 0
    for n in range(N) :
        for k in range(K) :
            J = J + r[n,k] * ((x0[n] - mu[k, 0])**2
                             + (x1[n] - mu[k, 1])**2)
    return J

# test
# Mu와 R의 초기화

Mu = np.array([[-2,1], [-2,0], [-2,-1]])
R = np.c_[np.ones((N,1), dtype=int), np.zeros((N,2), dtype=int)]

distortion_measure(X[:, 0], X[:, 1], R, Mu)

# 실행결과 : 771.7091170334878
```


	- 실행 결과의 771.어쩌구가 초기값에 의한 왜곡 척도이다.
	- 이제 이거로 반복에 의한 왜곡 척도를 구해보자.
	
	
```python
# 초기화 포함하여 시각화까지

N = X.shape[0]
K = 3
Mu = np.array([[-2,1], [-2,0], [-2,-1]])
R = np.c_[np.ones((N,1), dtype=int), np.zeros((N,2), dtype=int)]
max_it = 10
it = 0
DM = np.zeros(max_it) # 왜곡 척도의 계산 결과를 낳는다

for it in range(0, max_it) : # k-means 기법
    R = step1_kmeans(X[:, 0], X[:, 1], Mu)
    DM[it] = distortion_measure(X[:, 0], X[:, 1], R, Mu) # 왜곡 척도
    Mu = step2_kmeans(X[:, 0], X[:, 1], R)
print(np.round(DM, 2))

plt.figure(2, figsize=(4,4))
plt.plot(DM, color='black', linestyle='-', marker='o')
plt.ylim(40, 80)
plt.grid(True)
plt.show()
```


	- print 문에서 반복 척도(2까지 반복) 에 의한 왜곡 척도는 다음과 같다.
	
			[627.54  73.39  70.39  57.59  48.31  47.28  46.86  46.86  46.86  46.86]
		
R을 갱신한 후 왜곡척도 J를 구해봤는데,

	- 왜곡 척도 함수의 각 point 점이 J가 됨

왜곡 척도는 반복 계산을 통해 **계속 감소하는 것을** 알 수 있다.


> 예제 내용을 통해 k-means 의 단계를 정리해보면..
		
1. k-means 기법으로 얻을 수 있는 해는 초기값 의존성이 있다.
2. 처음 Mu에 무엇을 할당하는 지에 따라 결과가 달라질 수 있다.
3. 실제는 다양한 Mu에서 시작하여 얻은 결과중에 가장 왜곡척도가 작은 결과를 사용하는 방법이 쓰인다.
4. 또한 예제에서는 Mu를 먼저 정했지만 R을 먼저 결정해도 무방하다.
5. 이 경우, R을 임의로 정해 거기서 Mu를 찾아가는 절차이다.



### 그 외 다른 클러스터링 방식


##### K-medoids (객체, 대푯값)



K-medoids는 K-means 보다 robust.

	- noise, outliar에 덜 취약
	- medoid가 mean 보다는 극단적인 outliar에 덜 취약하기 때문

![K-medoids](https://www.researchgate.net/publication/282897075/figure/fig4/AS:325108624314429@1454523344524/k-medoids-clustering.png)



##### GMM (Gaussian Mixture Model)



데이터가 k개의 가우시안으로 구성되어 있다고 간주하고,

데이터를 가장 잘 설명하는 k개의 평균과 covariance 를 찾는 알고리즘

![GMM](https://scikit-learn.org/stable/_images/sphx_glr_plot_gmm_covariances_0011.png)


	- k-means 기법은 데이터 점을 반드시 클러스터에 할당한다
		- 예를 들어 클러스터 0의 중심에 있는 데이터 점 A도, 클러스터 0의 끝에 있는 데이터 점 B에도 "동일한" R=[1,0,0]이 할당
		
> 그럼 이런 경우에는 어떻게 하죠?

	- 데이터 점 A는 확실히 클러스터 0에 속하지만, 데이터 점 B는 클러스터 0과 클러스터 1에 모두 속해있으면?
	
GMM에서는 이런 이슈를 어떻게 해결하는지 알아보자.

	- 예를 들어, 데이터 점 A가 클러스터 0에 속할 확률은 0.9이며, 클러스터 1,2에 속할 확률은 각각 0.1과 0.0이라 하면,
	- 이것을 감마(Gamma)를 사용하여 다음과 같이 나타낸다
	
	- Gamman = [Gamma0, Gamma1, Gamma2] = [0.9, 0.1, 0.0]
	
	- 어떤쪽의 클러스터에는 반드시 속하므로, 3개의 확률을 더하면 1이다
	
	- K-means 에서의 R 의미가 확장된다.
	
	- 한편, 클러스터 0의 가장자리에 있던 데이터 점 B는, 
	- Gamman = [Gamma0, Gamma1, Gamma2] = [0.5, 0.4, 0.1]
	- 과 같이, 클러스터 0에 속할 가능성을 작은 수치로 나타낼 수 있다.
	

> 클러스터 k에 속할 확률의 의미?


	- 같은 종류의 곤충이라 생각하고, 여러 마리를 채취하여 질량과 크기의 데이터를 기록하고 200마리를 모아 플롯한 결과
	- 3개의 클러스터가 있는 것으로 나타났다고 일단 가정하자
	
	- 이 경우 외형은 같다고 생각해 수집한 곤충이 실은 '적어도 세 가지의 변종이 있었다' 라고 해석할 수 있음
	
	- 모든 곤충은 어떤 쪽의 변종에 속해있으며, 이에 따라 질량과 크기가 정해진다고 생각할 수 있다.
	- 3개의 클러스터의 뒤에 3개의 클래스의 존재가 암시된 것

이렇게 관찰은 못했지만 데이터에 영향을 준 변수를 **잠재변수** , 또는 **숨은변수 (Hidden variable)** 이라고 한다.


	- 이 잠재변수를 3차원 벡터를 사용하여 표현해보자. (Z라고 할게요)
	
	- Zn = [Z0, Z1, Z2]
	
	- 데이터 n이 클래스 k에 속한다면 Zk 만 1을 취하고 다른 요소는 0으로 함
		- 예를 들어, n번째 데이터가 클래스 0에 속해있으면, Zn = [1,0,0] 이고 클래스 1이면 Zn = [0,1,0]
	- 모든 데이터를 정리하여 행렬로 나타낼 땐 Z와 같이 대문자로 표기 (이건 K-means 기법의 R과 거의 똑같음)
	
	- 그런데, 이 관점에서 데이터 n이 클러스터 k에 속할 확률 GammaK 란,
	- 데이터 xn인 곤충이 클래스 k의 변종일 확률을 의미한다.
	

이 Gamma는 "어떤 클러스터에 얼마나 기여하고 있는가" 라는 의미에서 **부담률 (Responsibility)** 라고함


![GMM2](https://t1.daumcdn.net/cfile/tistory/272AF83353CEA9B215)

(참고 URL : https://iskim3068.tistory.com/52)

(참고 URL2 : https://brunch.co.kr/@gimmesilver/40)




##### DB Scan


밀도 기반 군집화 : 점이 세밀하게 모여있는 경우 (밀도) 를 계산하여 군집한다.

(어느 점을 기준으로 반경 x내에 점이 n개 이상 있으면 하나의 군집으로 인식한다.)

Density-Based, 특히 DB Scan이 가장 많이 사용된다.

	- 앞선 K-means 방식의 경우 군집간의 거리를 이용한다.


![DB Scan](https://i.stack.imgur.com/O2r3i.png)


> 그래서 이게 뭔소린데?


(참고 URL : https://bcho.tistory.com/1205)


![DB Scan2](https://t1.daumcdn.net/cfile/tistory/9930A63359E057BA1A)

위 그림에서 minPts = 4 라고 하면, 파란점 P를 중심으로 반경 epsilon 내에 점이 4개 이상 있으면 하나의 군집으로 판단할 수 있는데, 

아래 그림은 점이 5개가 있기 때문에 하나의 군집으로 판단이 되고, P는 **core point**가 된다. 


![DB Scan3](https://t1.daumcdn.net/cfile/tistory/996B8A3359E057BA27)

위 그림에서 회색점 P2의 경우 점 P2를 기반으로 epsilon 반경내의 점이 3개 이기 때문에, 

minPts=4에 미치지 못하기 때문에, 군집의 중심이 되는 core point는 되지 못하지만, 

앞의 점 P를 core point로 하는 군집에는 속하기 때문에 이를 **boder point (경계점)**이라고 한다. 


![DB Scan4](https://t1.daumcdn.net/cfile/tistory/997CC13359E057BA2F)

아래 그림에서 P3는 epsilon 반경내에 점 4개를 가지고 있기 때문에 core point가 된다.


![DB Scan5](https://t1.daumcdn.net/cfile/tistory/9937193359E057B934)

그런데 P3를 중심으로 하는 반경내에 다른 core point P가 포함이 되어 있는데, 

이 경우 core point P와  P3는 연결되어 있다고 하고 하나의 군집으로 묶이게 된다.


![DB Scan6](https://t1.daumcdn.net/cfile/tistory/99D7893359E057B938)

마지막으로 위 그림의 P4는 어떤 점을 중심으로 하더라도 minPts=4를 만족하는 범위에 포함이 되지 않는다. 

즉 어느 군집에도 속하지 않는 outlier가 되는데, 이를 **noise point**라고 한다. 

이를 모두 정리하면 아래와 같은 그림이 나온다.


![DB Scan7](https://t1.daumcdn.net/cfile/tistory/99CC563359E057BA25)



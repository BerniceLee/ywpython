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



	

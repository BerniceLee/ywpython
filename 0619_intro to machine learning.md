# 기계학습이란?


### 기계학습이 사용되는 예시는?


	* 스팸메일 분류
	* 추천시스템
	* 글자인식
	* 사물인식
	* 신용, 대출조회/판단
	* 부정사용 감지
	
> 기계학습의 핵심은 "분류기 (classifier)"
	
	* 정말 다양한 형태로 여러 형태의 선을 긋는 것
	* 선을 긋는 수많은 논리들 -> "기계학습"
  	
![Image of classifier](http://en.proft.me/media/science/ml_classification.jpg)
	
	
### 기계학습의 정의


##### ETP

	* Experience -> data
	* Task -> model
	* Performance Measure -> cost func
	
##### 기계학습의 유형
	
![Types of machine learning](http://en.proft.me/media/science/ml_types.png)

##### 기계학습의 논리

*descriptive VS predictive*

![Types of data analysis](https://insights.principa.co.za/hs-fs/hubfs/blog-files/4-types-of-data-analytics-principa.png?width=1101&height=609&name=4-types-of-data-analytics-principa.png)

* descriptive

	* "What's happening in my business?"
	* 설명적 분석 : 지나간 결과를 분석해서 어떤일이 발생했는지 쉽게 파악하고자 함
	
* predictive
	
	* "What's going to happen in future?"
	* 예상적 분석 : 향후 발생할 현상이나 결과를 예쌍하는 분석
	
![Predictive analysis](http://en.proft.me/media/science/ml_predictive_chain.jpg)

##### Workflow (supervised learning)

	* 머신러닝 기법은 predictive!
	
![Machine learning workflow](https://wikidocs.net/images/page/31947/%EB%A8%B8%EC%8B%A0_%EB%9F%AC%EB%8B%9D_%EC%9B%8C%ED%81%AC%ED%94%8C%EB%A1%9C%EC%9A%B0.PNG)

	머신러닝을 하는 과정은 크게 6가지 단계로 나눈다.
	
	1. 수집 (Acquisition)
	2. 점검 및 탐색 (Inspection and Exploration)
	3. 전처리 및 정제 (Preprocessing and Cleaning)
	4. 모델링 및 훈련 (Modeling and Training)
		
		* 전체 데이터중, 일부 데이터는 검증용, 일부 데이터는 테스트 용으로 분류한다.
		
![Data sorting](https://wikidocs.net/images/page/31947/%EB%8D%B0%EC%9D%B4%ED%84%B0.PNG)

	5. 평가 (Evaluation)
	6. 배포 (Deployment)
	
##### Model evaluation

	*모델 성능 평가*
	
	https://bit.ly/2ZyfBKe
	(위의 링크에 모델 성능평가에 대한 개요가 잘 나와있다!)
	
> 모델을 연구나 비즈니스에 적용할 때 공통으로 원하는 한가지는 바로 “좋은” 예측


### 기계학습에 쓰이는 주요 용어


##### train ＆ test

![Train and test](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Machine_learning_nutshell_--_Split_into_train-test_set.svg/768px-Machine_learning_nutshell_--_Split_into_train-test_set.svg.png)

	* 전체 sample data의 크기중, 일부는 훈련용, 일부는 테스팅용으로 분리한다.
	
##### validation

![validation](https://t1.daumcdn.net/cfile/tistory/9951E5445AAE1BE025)

> 그냥 training set 과 testing set 으로만 나누면 되지, 왜 굳이 validation set을 나눌까?
	
	* "모델의 성능 평가를 위함이다!"
	
		* 즉, training set 의 일부를 모델 성능평가를 위하여 사용한다.
		
			* 모델 완성 후, 실제 환경에서 잘 동작할지 가늠해볼 때, training set 을 사용하면 곤란하므로
			* 이 경우 validation set 을 사용해준다.
			
*K-fold validation*
	
![K-fold validation](https://cdn-images-1.medium.com/max/1600/1*rgba1BIOUys7wQcXcL4U5A.png)

		* 교차검증 이라고도 한다.
		* K개의 fold 를 만들어서 진행하는 교차검증
		
			* 총 데이터 갯수가 적은 데이터 셋에 대해 정확도를 향상시킬 수 있음
			* 기존에 training / validation / test 세 개의 집단으로 분류하는 것 보다, training / test 로 분류할 때 학습 데이터 set이 더 많기 때문
			* 데이터 수가 적은데 검증과 테스트에 데이터를 다 뺏긴다면, underfitting 되는 등, 성능이 미달되는 모델이 학습된다.
			
			1. training / test set 으로 나누고
			2. training 을 K개의 fold 로 나눈다
			3. 한 개의 fold에 있는 데이터를 다시 K개로 쪼개고, K-1개는 training, 마지막 하나는 validation set 으로 지정
			4. 모델 생성하고 예측 진행, 에러값 추출
			5. 다음 fold 로 넘어가면 validation set 을 바꿔서 지정하고, 이전 fold 에서 validation 역을 했던 set은 training set 으로 재활용
			6. 이것을 K번 반복한다.
			
			* 최적의 모델 찾기는 좋겠지만, 모델 학습 시간소요가 클 것 같다.
			
##### labeling (=classification)

![Labeling](https://blogs.nvidia.com/wp-content/uploads/2018/07/Supervised_machine_learning_in_a_nutshell.svg_-842x263.png)

##### 정오분류표 (contingency table)

![Contingency table](http://www.jangun.com/study/img/datam016.jpg)

	* 실제 범주 vs 모형에 의해 예측된 분류 범주의 관계를 나타낸 표
		
		
		

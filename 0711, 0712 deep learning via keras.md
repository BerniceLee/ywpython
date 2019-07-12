# 케라스를 이용한 딥러닝 개발

일반적으로 케라스를 사용하여 작업할 땐

1. 훈련 데이터 정의
2. 네트워크 정의
3. 손실 함수, 옵티마이저, 성능 지표 설정

이 순서대로 작업한다.

케라스에는 2가지 모델 정의 방식이 있다.

- Sequential 클래스 : 말 그대로 순차적으로 
- 함수형 (functional) API : 기능별로 정리


## 경사하강법의 종류


**배치 경사 하강법 (Batch Gradient Descent)**

매 반복마다 전체 훈련 데이터셋을 사용하여 가중치를 갱신한다

**확률적 경사 하강법 (Stochastic Gradient Descent, SGD)**

매 반복마다 한개의 샘플을 사용하여 가중치를 갱신한다

**미니 배치 확률적 경사 하강법 (Mini-Batch Stochastic Gradient Descent, mini-batch SGD)**

매 반복마다 mini-batch 단위로 가중치를 갱신한다.


> 좀 더 자세하게 살펴보자.


### ML 에서 이야기하는 Batch 란?


- 모델을 학습할 때 한 Iteration 당 사용되는 example set 모임
- 여기서 iteration 은 정해진 batch size를 이용하여 학습을 반복하는 횟수를 말한다.
- 한번의 epoch를 위해 여러번의 iteration 이 필요함


### Batch size의 정의 및 batch size를 선택하는 방법


- Batch 하나에 포함되는 example set의 갯수

[위 설명에 대한건 여기를 참고하자](https://nonmeyet.tistory.com/entry/Batch-MiniBatch-Stochastic-%EC%A0%95%EC%9D%98%EC%99%80-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EC%98%88%EC%8B%9C)

- SGD는 배치크기가 1, Mini-Batch는 10~1000 사이긴 한데, 보통 2의 지수승


이제 하나씩 다시 살펴보자.


### BGD (Batch Gradient Descent)


**전체 데이터셋** 에 대한 에러를 구하고, 기울기를 한번만 계산해서 model parameter를 업데이트 하는 방법.


BGD의 장점


● 전체 데이터에 대해 업데이트가 한번에 이루어지기 때문에 후술할 SGD 보다 업데이트 횟수가 적다. 따라서 전체적인 계산 횟수는 적다.

● 전체 데이터에 대해 error gradient 를 계산하기 때문에 optimal 로의 수렴이 안정적으로 진행된다.

● 병렬 처리에 유리하다.



BGD의 단점



● 한 스텝에 모든 학습 데이터 셋을 사용하므로 학습이 오래 걸린다.

● 전체 학습 데이터에 대한 error 를 모델의 업데이트가 이루어지기 전까지 축적해야 하므로 더 많은 메모리가 필요하다.

● local optimal 상태가 되면 빠져나오기 힘듦(SGD 에서 설명하겠음.)


### SGD (Stochastic Gradient Descent)


- **추출된 데이터 한 개에 대해서** error gradient 를 계산하고, Gradient descent 알고리즘을 적용하는 방법


SGD의 장점



● 위 그림에서 보이듯이 Shooting 이 일어나기 때문에 local optimal 에 빠질 리스크가 적다.

● step 에 걸리는 시간이 짧기 때문에 수렴속도가 상대적으로 빠르다.



SGD의 단점



● global optimal 을 찾지 못 할 가능성이 있다.

● 데이터를 한개씩 처리하기 때문에 GPU의 성능을 전부 활용할 수 없다.



### Mini-batch Gradient Descent (MSGD)


![mini-batch](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile26.uf.tistory.com%2Fimage%2F99BE27485D0F73FD2BAD11)

- **전체 데이터셋에서 뽑은 Mini-batch 안의 데이터 m 개에 대해서** 각 데이터에 대한 기울기를 m 개 구한 뒤, 그것의 평균 기울기를 통해 모델을 업데이트 하는 방법


MSGD의 장점



● BGD보다 local optimal 에 빠질 리스크가 적다.

● SGD보다 병렬처리에 유리하다.

● 전체 학습데이터가 아닌 일부분의 학습데이터만 사용하기 때문에 메모리 사용이 BGD 보다 적다.



MSGD의 단점



● batch size(mini-batch size) 를 설정해야 한다.

● 에러에 대한 정보를 mini-batch 크기 만큼 축적해서 계산해야 하기 때문에 SGD 보다 메모리 사용이 높다.



SGD 중에서도 저 종류만 있는게 아니라, 여러가지 변종들이 있다.


### SGD의 여러 변종들


- Adam, RMSProp, Adagrad 등 (이걸 optimizer 라고 부름)
- 모멘텀(momentum)
  - 현재의 gradient뿐만 아니라 (가속도로 인한) 속도를 고려하여 가중치 계산

모멘텀말고도 여러가지 있는데, 아래 참고 링크에서 확인이 가능함.

[경사하강법의 여러가지 알고리즘 정리](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html)
	
	
  
## 오차역전법


우리가 기존에 배웠던 것은, Input -> Output 으로 가중치를 업데이트 하면서 활성화 함수를 통해 결과값을 얻는 과정이다.

이렇게 쭉 오는것을 **순전파 (forward)** 라고 하는데, 임의로 한 번 순전파 했다고 출력 값이 정확하지는 않을 것이다.

오차역전파는 **결과값을 통해 다시 input, 역방향으로 오차를 다시 보내며 가중치를 재업데이트 하는 것**이다.


![순전파](https://i.stack.imgur.com/H1KsG.png)

위 그림은 순전파를 나타낸건데, 이 error 값을 다시 역방향으로 hidden layer, input layer 로 오차를 다시 보내서 가중치를 다시 계산한다.


![역전파](https://t1.daumcdn.net/cfile/tistory/997C7A3359EEF5CA1F)

이 그림을 보면... error 값이 0.6인데, 각 hidden layer 에 0.36, 0.24로 쪼개서 다시 나눠준다.

> 각 노드가 영향을 미친 만큼 오차를 차등하여 역전파 하는것이 맞는듯.



## 데이터 분리 및 전처리


데이터 분리를 할 때 몇가지 주의할 점이 있다.

- **데이터를 대표할 수 있게 구성할 것.**
	- 분리하기 전에 데이터가 잘 섞여 있는지 확인하자.

데이터를 분류할때, 그리고 중복되지 않게 잘 섞어줘야한다.


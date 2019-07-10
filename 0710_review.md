지금까지 했던 내용까지만 (~softmax) 나름대로 리뷰를 좀 해보고자 함.


## Linear Regression


**H(X) = Wx + b** 

기본 선형회귀방정식이다. 이걸 추출하기 위한 코드를 작성해서 보자.

우선 cost function 을 알아야하는데, 그걸 먼저 뽑아보면



```python
import tensorflow as tf
import matplotlib.pyplot as plt

X = [1,2,3]
Y = [1,2,3]

# placeholder : W값을 임의대로 바꿔가면서 경과를 보겠다는 의미
W = tf.placeholder(tf.float32)
# H(x) = Wx
hypothesis = X*W

cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 그래프를 그리기 위한 세션을 열어줌
sess = tf.Session()
# 그래프 전역변수 초기화
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []
for i in range(-30,50) :
    feed_W = i * 0.1
  # 0.1씩 움직이면서 변하는 모든 cost 값을 dictionary 안에 저장
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()
```

![cost function](https://engmrk.com/wp-content/uploads/2018/05/cost-function-with-one-optima.jpg)

이렇게 생긴 cost function 이 나옴.

이제 이 cost function에 경사하강법을 적용을 시켜보자.

![cost function2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile5.uf.tistory.com%2Fimage%2F99040D505A5E08EF153481)

cost function 의 각 기울기를 줄여나가기 위해 미분한 최종 cost function을 그대로 대입.

적용을 시키기 위한 코드는 아래와 같음


```python
learning_rate = 0.1
# gradient는 미분한 수식을 그대로 써주고
gradient = tf.reduce_mean((W*X-Y) * X)
# 미분값에 gradient x 를 곱한값을 W에서 빼주면 최종 수식 완성
descent = W - learning_rate * gradient
# assign = 새로운 변수값을 적용한다는 뜻
update = W.assign(descent)
```

이걸 적용해서 모델을 학습 시켜보면


```python
# 위 블록을 적용하면

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X*W

cost = tf.reduce_sum(tf.square(hypothesis-Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W*X-Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(21) :
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
```

```python
'''
0 23.546288 [-0.29687226]
1 6.69761 [0.30833483]
2 1.9050977 [0.6311119]
3 0.54189456 [0.8032597]
4 0.15413892 [0.8950718]
5 0.04384405 [0.9440383]
  ...
16 4.3203613e-08 [0.99994445]
17 1.2296329e-08 [0.9999704]
18 3.4985028e-09 [0.9999842]
19 9.918502e-10 [0.9999916]
20 2.7818103e-10 [0.9999955]
'''
```

근데 코드를 저렇게 쓰면 좀 복잡하니까, tensorflow 내에 누가 편하게 쓰라고 정의해놓은 것을 가져다가 쓰면

```python
optimizer = tf.train.GradientDescentOptimizer(learing_rate=0.1)
train = optimizer.minimize(cost)
```

이렇게 압축 가능.

이제 이걸 또 적용시키면


```python
X = [1,2,3]
Y = [1,2,3]

# 5.0으로 주면 -> 어느순간 1에서 W가 멈춤 : 1이 최적값
W = tf.Variable(5.0)

hypothesis = X*W

# reduce_sum 으로 해도 이 예제에서는 크게 상관이 없다. 근데 reduce_mean 이 원칙적으로는 맞는 말
cost = tf.reduce_mean(tf.square(hypothesis-Y))

# learning_rate = 0.1
# gradient = tf.reduce_mean((W*X-Y) * X)
# descent = W - learning_rate * gradient
# update = W.assign(descent)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100) :
#   sess.run(update, feed_dict={X: x_data, Y: y_data})
#   print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
    print(step, sess.run(W))
    sess.run(train)
```

```python
'''
0 5.0
1 1.2666664
2 1.0177778
3 1.0011852
4 1.000079
5 1.0000052
6 1.0000004
7 1.0
8 1.0
9 1.0
10 1.0
11 1.0
12 1.0
13 1.0
14 1.0
  ...
93 1.0
94 1.0
95 1.0
96 1.0
97 1.0
98 1.0
99 1.0
'''
```

Accuracy 가 1이되서 최종 W값에 도달했음

-3.0 정도에서 실행해도 똑같이 어쨌든 1.0 Accuracy 에 도달함


> 경사하강법의 장점 : 결국은 최적화된 Accuracy 1.0에 도달할 수 있음


하다보면, 저 경사값 gradient를 손보고 싶을 때가 생길 것임

그럴땐 바로 optimize 안하고, gradient를 계산한 값을 변수로 저장함

gvs 변수값을 조정하면 gradient 를 수정 가능함


```python
gvs = optimizer.compute_gradients(cost)
# 수정한 gradient 값을 apply_gradients 메소드로 적용해줌
apply_gradients = optimizer.apply_gradients(gvs)
```

이걸 이제 적용시켜보면



```python
X = [1,2,3]
Y = [1,2,3]

W = tf.Variable(5.0)

hypothesis = X*W

gradient = tf.reduce_mean((W*X-Y) * X) * 2

cost = tf.reduce_sum(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

gvs = optimizer.compute_gradients(cost)
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100) :
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)
```    

```python
'''
0 [37.333332, 5.0, [(112.0, 5.0)]]
1 [-67.200005, -6.2000003, [(-201.6, -6.2000003)]]
2 [120.960014, 13.960001, [(362.88004, 13.960001)]]
3 [-217.72803, -22.328003, [(-653.1841, -22.328003)]]
4 [391.9105, 42.990406, [(1175.7314, 42.990406)]]
5 [-705.43896, -74.58274, [(-2116.317, -74.58274)]]
6 [1269.7902, 137.04895, [(3809.3706, 137.04895)]]
7 [-2285.6226, -243.88812, [(-6856.8677, -243.88812)]]
8 [4114.1206, 441.79865, [(12342.362, 441.79865)]]
9 [-7405.418, -792.4376, [(-22216.254, -792.4376)]]
  ...
93 [-2.0532455e+25, -2.1999058e+24, [(-6.159737e+25, -2.1999058e+24)]]
94 [3.6958423e+25, 3.959831e+24, [(1.1087527e+26, 3.959831e+24)]]
95 [-6.652516e+25, -7.127696e+24, [(-1.9957549e+26, -7.127696e+24)]]
96 [1.19745305e+26, 1.2829853e+25, [(3.592359e+26, 1.2829853e+25)]]
97 [-2.1554157e+26, -2.309374e+25, [(-6.466247e+26, -2.309374e+25)]]
98 [3.8797484e+26, 4.1568732e+25, [(1.16392454e+27, 4.1568732e+25)]]
99 [-6.983548e+26, -7.4823725e+25, [(-2.0950643e+27, -7.4823725e+25)]]
```

저 결과 코드를 좀 분석해보면,

- 첫번쨰엔 텐서플로우에서 계산한 값, 그 다음은 w, 그 다음은 [수정한 gradient, weight 값] 
- 이걸 쭉 돌리면서 보면, 텐서에서 계산한 gradient 와 수정해서 계산한 gradient 값이 거의 유사한걸 볼 수 있음
- 고로 저렇게 변화를 줘서 사용해도 무방하다.


> 지금껀 variable 이 단일이여서 굉장히 쉬운 예제였고, 여러가지 variable 인 경우는?



## Multi-variable Linear Regression


![multi-variable ex](https://t1.daumcdn.net/cfile/tistory/99DBEA505A71A25610)

수업시간에 했던 저 예제 그대로 쓰자.

이걸 토대로 모델을 학습시켜 6번째 학생의 점수를 유추하고자 함.

> 모델 학습 시킬때마다 너무 난수가 왔다리 갔다리 인데, 이걸 고정시키려면?

```python
tf.set_random_seed(777)
```

**set_random_seed** 를 쓴다.

이걸 쓴다고 매 학습회차마다 값이 고정이 되는 것은 아니지만,
난수 변역대를 그래도 고정시켜서 여기 저기서 코드 다 실행시킬때마다 고정된 난수 영역대가 나올수 있게 잡아줌.


```python
import tensorflow as tf
tf.set_random_seed(777)

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# learning rate = 0.0001
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
```

```python
'''
0 Cost:  19614.8
Prediction:
 [ 21.69748688  39.10213089  31.82624626  35.14236832  32.55316544]
10 Cost:  14.0682
Prediction:
 [ 145.56100464  187.94958496  178.50236511  194.86721802  146.08096313]
 ...
1990 Cost:  4.9197
Prediction:
 [ 148.15084839  186.88632202  179.6293335   195.81796265  144.46044922]
2000 Cost:  4.89449
Prediction:
 [ 148.15931702  186.8805542   179.63194275  195.81971741  144.45298767]
'''
```

Prediction 값이 Cost 순서가 올라갈수록 우리가 추론하고자 할 Y 값과 상당히 유사하게 나타남



## Logistic Regression



![sigmoid](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/480px-Logistic-curve.svg.png)

시그모이드 함수를 쓴다.

쉽게 말하면 이진분류인데, 0 아니면 1로 분류되는 상황에서 쓰이는 방법.

시그모이드 함수는 0 아니면 1로 분류를 하는거라서 cost function 이 linear regression 처럼 매끄럽게 안나오고,
약간 울퉁 불퉁하게 나오게 된다.

![sigmoid2](https://t1.daumcdn.net/cfile/tistory/245FCB48579821A913)

이런식으로 되는데,

저 울퉁불퉁한 면을 줄이기 위해 시그모이드 함수와 상극인 log를 맥임.

![sigmoid3](https://t1.daumcdn.net/cfile/tistory/27623248579821A911)

그래서 시그모이드의 cost function은 log를 품은 꼴이 된다.

이걸 토대로, cost function을 재구성 시키면..
   


- **cost(W) = (1/m) ∑ c(H(x), y) 의 재구성**
- **(1/m) ∑  -(ylog(H(x)) + (1-y)log(1-H(x)))   ==>   -(1/m) ∑ylog(H(x)) + (1-y)log(1-H(x))**

   


이제 텐서플로우 코드를 통해 적용시켜보자.


```python
import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
# y-data 는 binary 로 주어짐
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# W -> x는 두개의 feature, y는 하나의 feature
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 시그모이드 함수 : tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 정확도 계산
# 0.5 이상이면 True, 아니면 False
# cast는 형변환을 해주는 메소드
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        # 200번 정도마다 한번씩 cost 를 출력해보자.
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
``` 

```python
'''
0 1.73078
200 0.571512
400 0.507414
600 0.471824
800 0.447585
...
9200 0.159066
9400 0.15656
9600 0.154132
9800 0.151778
10000 0.149496
Hypothesis:  [[ 0.03074029]
 [ 0.15884677]
 [ 0.30486736]
 [ 0.78138196]
 [ 0.93957496]
 [ 0.98016882]]
Correct (Y):  [[ 0.]
 [ 0.]
 [ 0.]
 [ 1.]
 [ 1.]
 [ 1.]]
Accuracy:  1.0
'''
```


> 시그모이드 함수를 쓸 때, 이런 오류가 나타나기 쉬움


![sigmoid2](https://t1.daumcdn.net/cfile/tistory/245FCB48579821A913)


이 그림을 다시 보면, 울퉁불퉁한 면이 중간중간마다 평평하게 생겼는데,

저 부분을 **local-minimum** 이라고 한다.

(그 지역에서만 그 바운더리 내에서만 작은 값이라는 뜻)

우리가 최종적으로 H(X)에서 뽑아야 하는 값은 **global-minimum** 인데, local-minimum 에서 머무르는 경우가 생길 수 잇다.




## Softmax



딱 정확하게 내가 추구하고자 하는 예측 결과가 0 아니면 1 로 분류가 되는게 아니라, 여러가지 가짓수로 분류가 되는 상황들이 실제로는 더 많을것임.

![softmax](https://t1.daumcdn.net/cfile/tistory/233484395797F43B0A)


저런 상황처럼...

- A를 받는 학생도 있고 B를 받는 학생도 있고 C를 받는 학생도 있고
- 시그모이드를 세개를 쓰면 사실 되긴 된다. (위 그림 처럼)
- 근데 저러면 연산이 복잡해짐


그래서, 여러가지 가짓수로 분류하기 위해 이 로직을 한군데 다 때려넣은게 softmax다.

![softmax2](https://t1.daumcdn.net/cfile/tistory/214D6A4F5797F9C131)


A를 받을 확률이 0.7, B를 받을 확률이 0.2, C를 받을 확률이 0.1 이라고 분류함.

**중요한건, 모든 확률의 합은 1이 되야한다.**

그래서 그림처럼 예측한 결과값은 1개가 아니라 **3개가 된다.**


![softmax3](https://t1.daumcdn.net/cfile/tistory/25582B4F5797F9C12D)


최종 그림으로 나타내면 위와 같음.


정리하면, 소프트맥스 함수는 두 가지를 수행한다.

- **1. 입력을 sigmoid와 마찬가지로 0과 1 사이의 값으로 변환한다.**
- **2. 변환된 결과에 대한 합계가 1이 되도록 만들어 준다.**


![softmax4](https://t1.daumcdn.net/cfile/tistory/2353D14F5797F9C22E)

저 그림에서 보면, **One-Hot-Encoding** 이라는게 나오는데,

가장 큰 값( A=0.7 )을 1로 치고 나머지는 다 0으로 친다는 의미다.

이거는 그냥 파이썬에서 **argmax()** 를 쓰면 된다. (최대값 추출해주는 메소드)


- 최종적으로 Cost function 을 뽑아보면 다음과 같다.


![softmax5](https://t1.daumcdn.net/cfile/tistory/2153C64F5797F9C32E)

여러가지 데이터들이 들어갈 때, 최대값만 1, 나머지는 0 으로 여러가지 예측값들이 나온다.


> cross-entropy 는 무슨의미?

- entropy는 열역학에서 사용하는 전문 용어로 복잡도 내지는 무질서량을 의미한다. 
- 엔트로피가 크다는 것은 복잡하다는 뜻이다. 
- cross-entropy는 통계학 용어로, 두 확률 분포 p와 q 사이에 존재하는 정보량을 계산하는 방법을 말한다. 

별 큰 의미가 이름에 숨어있진 않은듯.


![softmax6](https://t1.daumcdn.net/cfile/tistory/243B804E579806C11E)

예쁘게 정리해보면 저렇게 됨.

예측이 맞으면 cost 가 0, 아니면 무한대로 나온다 (!)

- 왜 무한대로 나오냐면....
  - 예측값이 맞으면 1 이라고 하자. 그걸 시그모이드의 기본 cost fucntion 뼈대인 -log(z)에 넣으면
    - cost 값이 0이됨.
  - 반대로 0이면, -log(z)에 넣으면 무한대로 발산함.
  
  

이걸 이제 코드로 구현해보면...


```python
import tensorflow as tf
import numpy as np

# softmax이기 때문에 y를 표현할 때, 벡터로 표현한다.
# 1개의 값으로 표현한다고 할 때, 뭐라고 쓸지도 사실 애매하다.

# 05train.txt
# #x0 x1 x2 y[A   B   C]
# 1   2   1   0   0   1     # C
# 1   3   2   0   0   1
# 1   3   4   0   0   1
# 1   5   5   0   1   0     # B
# 1   7   5   0   1   0
# 1   2   5   0   1   0
# 1   6   6   1   0   0     # A
# 1   7   7   1   0   0

xy = np.loadtxt('05train.txt', unpack=True, dtype='float32')

# xy는 6x8. xy[:3]은 3x8. 행렬 곱셈을 하기 위해 미리 transpose.
x_data = np.transpose(xy[:3])
y_data = np.transpose(xy[3:])

print('x_data :', x_data.shape)     # x_data : (8, 3)
print('y_data :', y_data.shape)     # y_data : (8, 3)

X = tf.placeholder("float", [None, 3])  # x_data와 같은 크기의 열 가짐. 행 크기는 모름.
Y = tf.placeholder("float", [None, 3])  # tf.float32라고 써도 됨

W = tf.Variable(tf.zeros([3, 3]))       # 3x3 행렬. 전체 0.

# softmax 알고리듬 적용. X*W = (8x3) * (3x3) = (8x3)
hypothesis = tf.nn.softmax(tf.matmul(X, W))

# cross-entropy cost 함수
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

learning_rate = 0.01
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            feed = {X: x_data, Y: y_data}
            print('{:4} {:8.6}'.format(step, sess.run(cost, feed_dict=feed)), *sess.run(W))

    print('-------------------------------')

    # 1은 bias로 항상 1. (11, 7)은 x 입력
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7]]})
    print("a :", a, sess.run(tf.argmax(a, 1)))         # a : [[ 0.68849683  0.26731509  0.04418806]] [0]

    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4]]})
    print("b :", b, sess.run(tf.argmax(b, 1)))         # b : [[ 0.2432227   0.44183081  0.3149465 ]] [1]

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
    print("c :", c, sess.run(tf.argmax(c, 1)))         # c : [[ 0.02974809  0.08208466  0.8881672 ]] [2]

    # 한번에 여러 개 판단 가능
    d = sess.run(hypothesis, feed_dict={X: [[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
    print("d : ", *d, end=' ')
    print(sess.run(tf.argmax(d, 1)))                   # d :  ...  [0 1 2]
```

```python
'''
1200 0.780959 [-1.06231141 -0.26727256  1.32958329] [ 0.06808002 -0.11823837  0.05015869] [ 0.17550457  0.23514736 -0.41065109]
1400 0.756943 [-1.19854832 -0.29670811  1.49525583] [ 0.0759144  -0.11214781  0.0362338 ] [ 0.19498998  0.23733102 -0.43232021]
1600 0.735893 [-1.32743549 -0.32218221  1.64961684] [ 0.08333751 -0.10558002  0.02224298] [ 0.21336637  0.23823628 -0.45160189]
1800 0.717269 [-1.44994986 -0.34407791  1.79402602] [ 0.09020084 -0.09902247  0.00882212] [ 0.23099622  0.2384187  -0.46941417]
2000 0.700649 [-1.56689751 -0.36275655  1.92965221] [ 0.09643653 -0.09271803 -0.00371794] [ 0.248116    0.23818409 -0.48629922]
-------------------------------
a : [[ 0.68849683  0.26731509  0.04418806]] [0]
b : [[ 0.2432227   0.44183081  0.3149465 ]] [1]
c : [[ 0.02974809  0.08208466  0.8881672 ]] [2]
d :  [ 0.68849683  0.26731509  0.04418806] [ 0.2432227   0.44183081  0.3149465 ] [ 0.02974809  0.08208466  0.8881672 ] [0 1 2]
'''
```


지금 강사님이 구현해서 보여주는 코드는, 위에 인터넷상에 있는 것보다 더 간소화된 내용 같음.

강사님의 코드는 별도로 ipynb에 주석을 달아 리뷰함.

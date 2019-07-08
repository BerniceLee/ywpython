# 딥러닝 개요

## 딥러닝의 개발환경 구축

### 기초 개념들


### 퍼셉트론(Perceptron)


![퍼셉트론](https://image.slidesharecdn.com/lecture29-convolutionalneuralnetworks-visionspring2015-150504114140-conversion-gate02/95/lecture-29-convolutional-neural-networks-computer-vision-spring2015-9-638.jpg?cb=1430740006)

- 다수의 신호를 입력받아 하나의 신호를 출력함
- 뉴런의 수상돌기나 축색돌기처럼 신호를 전달하는 역할을 퍼셉트론에선 weight 가 담당함
- 가중치가 포함된 신호의 총합이 정해진 임계값(theta)을 넘었을 때 1 출력.
- 못 넘으면 0 또는 -1 출력

![퍼셉트론2](http://www.saedsayad.com/images/Perceptron_3.png)

> 퍼셉트론의 출력 값은 +-1 또는 0 이기 때문에 "선형 분류" 이다.


### GPU


GPU는 컴퓨터 하드웨어의 데이터를 읽어들여 연산처리를 하는 CPU의 기능과 비슷하다.

하지만 내부 구조는 CPU와 차이가 있다.

- CPU : 명령어가 입력된 순서대로 데이터를 처리하는 직렬(순차) 처리 방식에 특화되어있음.
  - 따라서, 한번에 한가지 명령어만 처리하기 때문에 연산 담당 ALU가 많을 수 밖에 없음
- GPU : 여러 명령어를 동시에 처리하는 병렬 처리 방식을 가지고 있음
  - 캐시 메모리의 비중이 크지 않고 ALU가 1코어에 수백 수천개가 달려있음
  
![CPU vs GPU](https://mblogthumb-phinf.pstatic.net/MjAxNzExMzBfMTk1/MDAxNTEyMDA4ODU3MTY3.xzZQxp4NpF4Pwd3A2LHxm_UcYYLGMZ3AwaXrzghtCgAg.MAht3GgxGoidFjt7LMtOeYCb9t-bZTPQDTOfgeod35Ig.PNG.suresofttech/image.png?type=w800)

> GPU와 딥러닝의 관계는?

원래 GPU가 게임쪽에서 주로 쓰였는데, 딥러닝에서 학습훈련을 시킬 때 많은 데이터를 이용해서 학습시키는게 필요해짐.

그래서 병렬 연산이 가능한 GPU를 딥러닝의 수치계산에 쓰곤 한다.


[GPU의 병렬 프로그래밍 언어 CUDA](https://blogs.nvidia.co.kr/2018/01/16/cuda-toolkit/)



### 텐서플로우(Tensorflow)


- 구글이 만든 머신러닝용 오픈 소스 라이브러리
- 정의된 계산 그래프 사이에서의 텐서들의 흐름을 의미함
- Define and Run 방식


### 케라스(Keras)


- 거의 모든 종류의 딥러닝 모델을 간편하게 만들고 훈련시킬 수 있는 파이썬용 high-level 딥러닝 프레임워크

특징

- 쉬운 API : 프로토타입 빠른 제작 가능
- MIT License : 상업적인 프로젝트에도 이용가능
- 다양한 백엔드 엔진을 기반으로 유연한 개발 가능


이번 모듈을 위해 새로운 가상환경을 만들어줬고, keras와 keras-gpu를 설치해줬다.

```python
"""
  CUDA 9.0
  cuDNN for CUDA 9.0
  python 3.6
"""  
conda install keras
conda install keras-gpu

# 별도로 pip install tensorflow-gpu 를 안해줘도 된다.
```

이거 하다가 자꾸 설치가 안되고 에러 뜨면

```python
conda config --set ssl_verify no
```

이거 해주고 설치하면 설치된다.

100% 설치가 안된 모듈 (나같은 경우는 cuDNN 모듈만 설치가 자꾸 안되었음) 만 따로 설치하려면

```python
conda install -c anaconda cudnn
```

이런식으로 개별적으로 설치하면 100% 설치가 완료된다.

인터프리터에서도 import 가 잘된다.


> 여기 강사님 후배가 운영하신다는(?) 블로그가 있는데 여기서 꿀팁을 알려주심

[prlab 블로그](https://smprlab.tistory.com/)

[폴더에서 바로 오른클릭으로 CMD 열기](https://smprlab.tistory.com/24)

위에 링크를 보면, 각 폴더에서 바로 오른클릭하면 그 디렉토리로 CMD가 실행된다.

![CMD창 바로열기](https://t1.daumcdn.net/cfile/tistory/995F27485AAA133631)



### 텐서플로우를 위한 준비



**텐서(tensor)란?**

텐서는 **N차원 배열** 을 의미한다.

![what is tensor?](https://i.stack.imgur.com/Lv1qU.jpg)

![graph](https://i.stack.imgur.com/NvQN8.png)


이런식으로 각 차원이 올라갈때마다 weight (가중치) 를 부여한다.

![tensor](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/Images/Tensor_1.png)

그니깐 대략 이런식이다.


> 딥러닝에서는 보통 4D까지의 텐서를 다룬다.


- 벡터 데이터
  - 2D 텐서 => (samples, features)
- 시계열(time-series) 또는 시퀀스(sequence) 데이터
  - 3D 텐서 => (samples, timesteps, features)
- 이미지
  - 4D 텐서 => (samples, height, width, channels) 또는 (samples, channels, height, width)
- 옹영상
  - 5D 텐서 => (samples, frames, height, width, channels) 또는 (samples, channels, frames, height, width)
  
  
> 그래서 선형대수학이 도대체 어디에 쓰이길래?


(참고 URL : https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/linear_algebra.html)

![Practical Ex1](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/Images/Linear_1.jpg)

![Practical Ex2](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/Images/BiasTrick.jpeg)

선형대수의 지식을 이용해 weights, bias 를 분류기를 single matrix 로 변형이 가능하다.

이런식으로도 응용이 가능함.


#### 맛보기 실습


[강사님 깃-numpy의 여러 모듈들~퍼셉트론 기본 연산까지](https://github.com/tjrjsgk/temp/blob/master/0_tensor-manipulation.ipynb)

각 모듈에 대한 설명과 자세한 부분은 오늘자 ipynb 참조

실습시간에 한 개념중 몇가지만 개념적으로 짚고 넘어가면,


**브로드 캐스팅(Broadcasting)**


행렬에서의 브로드캐스팅은, 두 행렬 A,B중 크기가 작은 행렬을 크기가 큰 행렬과 모양(shape)이 맞게끔 늘려주는 것을 의미한다.

예를 들어, 아래의 행렬처럼 (3, 3)행렬에 (1, 3)행렬을 더하려고 할 때, (1, 3)행렬을 (3, 3)처럼 확장시켜 주는 것이 바로 브로드캐스팅(Broadcasting)

![행렬에서의 브로드캐스팅](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile24.uf.tistory.com%2Fimage%2F23547D3758A401D8344933)


> 행렬 확장은 되는데, 행렬 축소는 데이터 손실때문에 불가능하다.


import keras 를 jupyter notebook 에서 하다보니 에러가 발생한다.

```python
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
C:\ProgramData\Anaconda3\envs\ywkeras\lib\site-packages\tensorflow\python\platform\self_check.py in preload_check()
     74         try:
---> 75           ctypes.WinDLL(build_info.cudart_dll_name)
     76         except OSError:

C:\ProgramData\Anaconda3\envs\ywkeras\lib\ctypes\__init__.py in __init__(self, name, mode, handle, use_errno, use_last_error)
    347         if handle is None:
--> 348             self._handle = _dlopen(self._name, mode)
    349         else:

OSError: [WinError 126] 지정된 모듈을 찾을 수 없습니다

During handling of the above exception, another exception occurred:

ImportError                               Traceback (most recent call last)
<ipython-input-47-3d1e6d42ad48> in <module>
----> 1 import tensorflow as tf
      2 from tensorflow import keras

C:\ProgramData\Anaconda3\envs\ywkeras\lib\site-packages\tensorflow\python\platform\self_check.py in preload_check()
     80               "environment variable. Download and install CUDA %s from "
     81               "this URL: https://developer.nvidia.com/cuda-90-download-archive"
---> 82               % (build_info.cudart_dll_name, build_info.cuda_version_number))
     83 
     84       if hasattr(build_info, "cudnn_dll_name") and hasattr(

ImportError: Could not find 'cudart64_100.dll'. TensorFlow requires that this DLL be installed in a directory that is named in your %PATH% environment variable. Download and install CUDA 10.0 from this URL: https://developer.nvidia.com/cuda-90-download-archive
```


10.0을 깔라고해서... 그냥 cuda 10.0로 업글해서 깔아줬다.

(아마 가상환경에서 install 진행할 때, CUDA 9.0 버전과 호환되는 패키지를 설치를 안하고 그냥 최신버전으로 깔아서 그런것 같음)

모듈 임포트 잘됨.

만약에 재부팅 하기 싫으면 그냥 환경변수 PATH 에 10.0 버전 설치경로 추가하면 되긴 된다.


### 퍼셉트론을 활용한 XOR 문제
  
  
![논리회로](https://study.com/cimages/multimages/16/bf424dd7-26f4-4634-8807-8e714f6cdb86_picture3.png)


가장 기본적인 구조.

[퍼셉트론을 이용한 기본 논리구조 연산](https://needjarvis.tistory.com/181)


> AND 게이트 같은 선형 논리는 계산이 가능한데, XOR 같은 비선형 논리는 풀이가 불가



  
  
  
  

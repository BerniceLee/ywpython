# 텍스트 분석

## 한국어 형태소 분석

### 형태소 분석

- 자연어의 문장을 "형태소" 라는 의미를 갖는 최소 단위로 분할하고, 품사를 판별하는 작업
  - 의미를 가지는 요소로서는 더 이상 분석할 수 없는 가장 작은 말의 단위
- 영어는 형태소마다 띄어쓰기로 하여 문장을 구성하기 때문에 어렵지 않다
- 아시아 언어 분석은 문법 규칙에 의한 방법과 확률적 언어 모델을 사용하는 방법 등을 사용한다.

**한국 형태소 분석 라이브러리**

- 다양한 형태소 분석 라이브러리가 오픈소스로 제공됨
- 파이썬에선 **KoNLPy** 를 사용

  - pos() : 형태소 분석
    - norm 옵션 : 단어 변환
      - 예를 들어 "그래욬ㅋㅋㅋㅋㅋ" -> 그래요 로 변환
    - stem 옵션 : 원형 단어 변환
      - 예를 들어 "그래욬ㅋㅋㅋㅋㅋ" -> 그렇다 로 변환


```python
# 한국어 분석 KoNLPy 사용

from konlpy.tag import Twitter

# Twitter 객체 생성
twitter = Twitter()

# pos 메소드로 형태소를 분석한다.
malist = twitter.pos("아버지 가방에 들어가신다." ,norm = True, stem = True)
print(malist)
```

```python
실행 결과 : 
[('아버지', 'Noun'), ('가방', 'Noun'), ('에', 'Josa'), ('들어가다', 'Verb'), ('.', 'Punctuation')]
```

> koNLPy 설치가 어렵다면....


내가 시도해봤던 것들 (별에 별거 다 해도 안되었는데 이렇게 하면 됨)

1. JVM 홈페이지 가서 최신버전 설치 (이 노트북은 64bit OS라서 64bit 로 깜)
2. (자바 설치하고 설정해도 되긴 됨) 환경변수 설정
  - 시스템 설정 환경변수 설정하는데 가서 "사용자 변수" 에서 새로 만들기 > JAVA_HOME, jdk 설치 디렉토리 추가하고
  - PATH 에서 "%JAVA_HOME%bin" 을 맨 앞에 추가하고
  - cmd 관리자 모드로 새로 켜서 java -version, javac -version 두개 다 제대로 나오는지 확인
3. 운영체제 버전도 64면 파이썬 버전도 64비트인지 확인

```python
import platform
print(platform.architecture())
```

4. 파이썬도 64bit 라서, JPype1-0.7.0-cp36-cp36m-win_amd64.whl 파일을 Anaconda3 파일 안으로 옮겨줌
5. pip upgrade -- 하라는대로 해주고, 그 다음 pip install JPype1.....whl
6. pip install konlpy
7. 주피터 커널 완전히 닫고 재시작하면 됨

8. from konlpy.tag import Twitter 호출할 때 윗줄에 다른 코드있으면 잘 안되서 가급적이면 첫줄로 놓기


**출현 빈도 분석하기**


각 형태소별로 빈도수를 측정해보자.

자료는 아래 데이터를 참조

**국립어학원 언어 정보 나눔터 말뭉치 데이터베이스**

https://ithub.korean.go.kr/user/total/database/corpusManager.do


```python
import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Twitter

# utf-16 인코딩으로 파일을 열고 글자를 출력한다.
fp = codecs.open("C:/Users/Affinity/Downloads/module4/ch02/BEXX0003.txt",
                "r", encoding = "utf-16")
soup = BeautifulSoup(fp, "html.parser")
body = soup.select_one("body > text")
text = body.getText()

# 텍스트를 한 줄씩 처리하기
twitter = Twitter()
word_dic = {}
lines = text.split("\n")

for line in lines :
    malist = twitter.pos(line)
    for word in malist :
        if word[1] == "Noun" :
            if not (word[0] in word_dic) :
                word_dic[word[0]] = 0
            word_dic[word[0]] += 1
            
# 많이 사용될 명사 출력
keys = sorted(word_dic.items(), key = lambda x: x[1], reverse = True)
for word, count in keys[:50] :
    print("{0}({1})\t".format(word, count), end="")
print()    
```

```python
실행 결과 :

것(644)	그(554)	말(485)	안(304)	소리(196)	길(194)	용이(193)	눈(188)	놈(180)	내(174)	사람(167)	봉(165)	치수(160)	평산(160)	얼굴(156)	거(152)	네(151)	일(149)	이(148)	못(147)	댁(141)	생각(141)	때(139)	강청댁(137)	수(134)	서방(131)	집(131)	나(122)	더(120)	서희(119)	머(116)	어디(112)	마을(111)	최(110)	년(109)	김(99)	칠성(97)	구천이(96)	니(96)	뒤(91)	제(90)	날(90)	아이(88)	하나(84)	녀(83)	두(83)	참판(82)	월(82)	손(81)	임(79)	
```


### 워드 클라우드


- 단어들을 구름 모양으로 나열하여 시각화
- 빈도가 높고 핵심단어 일수록 가운데에 크게 표현
- 영문은 **wordcloud** 패키지 사용, 한글은 **KoNLPy** 패키지에서 명사 추출하고 그다음 wordcloud 패키지


```python
# 워드클라우드

import matplotlib.pyplot as plt
%matplotlib inline

from wordcloud import WordCloud

text = open("C:/Users/Affinity/Downloads/module4/ch02/constitution.txt").read()
# 단어별 빈도 계산
wordcloud = WordCloud().generate(text)
wordcloud.words_
```

```python
실행 결과 : 

{'State': 1.0,
 'United States': 0.8181818181818182,
 'Law': 0.5151515151515151,
 'may': 0.5,
 'Congress': 0.4393939393939394,
 'President': 0.3939393939393939,
 'Section': 0.3333333333333333,
 'Person': 0.3333333333333333,
 'Office': 0.3333333333333333,
 'Year': 0.3181818181818182,
 'time': 0.30303030303030304,
 'House': 0.2878787878787879,
 'one': 0.2878787878787879,
 'Case': 0.2878787878787879,
 'Senate': 0.25757575757575757,
 'Power': 0.24242424242424243,
 'Constitution': 0.21212121212121213,
 'Vote': 0.21212121212121213,
 'Legislature': 0.19696969696969696,
 'thereof': 0.18181818181818182,
 ....
```

```python
plt.figure(figsize=(12,12))
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()
```


주어진 알파 이미지를 마스크로 활용하여 워드 클라우드를 만들 수 있다.


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
from wordcloud import WordCloud
from wordcloud import STOPWORDS

alice_mask = np.array(Image.open("C:/Users/Affinity/Downloads/module4/ch02/alice_mask.png"))

# 삭제할 단어 추가
stopwords = set(STOPWORDS)
stopwords.add("said")

# 텍스트 파일 임포트
text = open("C:/Users/Affinity/Downloads/module4/ch02/constitution.txt").read()
wordcloud = WordCloud(background_color="white", max_words=2000,
                      mask=alice_mask, stopwords=stopwords, max_font_size=40).generate(text)
```                      

워드클라우드를 한글 버전으로도 출력해보자.


```python
# 워드 클라우드 

import nltk
from konlpy.corpus import kolaw
from konlpy.tag import Twitter
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator

t = Twitter()

# 텍스트 파일 불러오기
ko_con_text = open(".../speech.txt",
                  encoding = "utf-8").read()
ko_con_text

# 명사 추출
tokens_ko = t.nouns(ko_con_text)
tokens_ko

# 단어 삭제
stop_words = []
f = open(".../stop_word.txt",
        encoding = "utf-8")
lines = f.readlines()
for x in lines :
  stop_words.append(x.strip())
stop_words
tokens_ko = [each_word for each_word in tokens_ko if each_word not in stop_words]
tokens_ko

sel_word = nltk.Text(tokens_ko)
data = sel_word.vocab().most_common(1000)
tmp_data = dict(data)

# 단어별 빈도 계산 (공백으로 분리한 단어)
wordcloud = WordCloud(font_path = "C:/Windows/Fonts/HYBDAM.ttf",
                      background_color = "white")
wordcloud = wordcloud.generate_from_frequencies(tmp_data)

# 시각화
plt.figure(figsize=(12,12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

```python
# 이미지 불러오기
korea_color = np.array(Image.open(".../south-korea-flag.png"))
image_colors = ImageColorGenerator(korea_color) # 이미지에서 색 추출

# 단어별 빈도 계산 (공백으로 분리한 단어)
wordcloud = WordCloud(font_path = "C:/Windows/Fonts/HYBDAM.ttf", relative_scaling = 0.2,
                     mask = korea_color, background_color = "white",
                     min_font_size = 1, max_font_size = 60)
wordcloud = wordcloud.generate_from_frequencies(tmp_data)

# 시각화
plt.figure(figsize=(12,12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.imshow(wordcloud, recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()
```


### 문장 -> 벡터 변환


**Word2Vec**


- 문장 내부의 단어를 벡터로 변환 (Embedding)
- 단어의 연결을 기반으로 단어의 연관성을 벡터로 생성하여 단어의 의미를 파악함

![Word2Vec](https://www.samyzaf.com/ML/nlp/word2vec2.png)


- Word2Vec의 결과를 2차원 그래프로 작성한다.

파이썬에서는 **genism** 라이브러리를 이용함

- 말뭉치(corpus)라고 불리는 단어사전을 생성하여 데이터를 준비하고 학습
- Twitter 형태소 분석기로 형태소를 나누고, Word2Vec 로 읽어들여서 데이터를 생성함


```python
import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Twitter
from gensim.models import word2vec

# utf-16 인코딩으로 파일 열고 글자를 출력하기
fp = codecs.open(".../BEX0003.txt",
                "r", encoding="utf-16")
soup = BeautifulSoup(fp, "html.parser")
body = soup.select_one("body > text")
text = body.getText()

# 텍스트를 한줄씩 처리한다.
twitter = Twitter()
results = []
lines = text.split("\r\n")
for line in lines :
  # 형태소 분석 - 단어의 기본형을 사용한다.
  malist = twitter.pos(line, norm = True, stem = True)
  r = []
  for word in malist :
    # 어미 / 조사 / 구두형 등은 대상에서 제외시킴
    if not word[i] in ["Josa", "Eomi", "Punctuation"] :
      r.append(word[0])
  r1 = (" ".join(r)).strip()
  results.append(r1)
  print(r1)
  
# 파일로 출력하기
wakati_file = "toji.wakati"
with open(wakati_file, "w", encoding="utf-8") as fp :
  fp.write("\n",join(results))
  
# Word2Vec 모델 만들기
data = word2vec.LineSentence(wataki_file) # 텍스트 읽기
model = word2vec.Word2Vec(data, size=200, window=10, hs=1, min_count=2, sg=1)
model.save(".../toji_model")
print("ok")
```

```python
model = word2vec.Word2Vec.load(".../toji_model")
model.most_similiar("땅") # 선택한 단어와 유사한 단어
model.similiarity("땅', "조상") # 두 단어 사이의 유사도 파악 (0~1사이값)

# (땅+조상+젊은이) 선형결과
model.most_similiar(positive=["땅", "조상"], negative=["젊은이"], topn=2)
```


## 베이즈 정리로 텍스트 분류하기


### 베이지안 필터

- 베이즈 정리를 이용한 텍스트 분류 방법으로 지도학습에 속함
- 여기선 **나이브 베이즈 알고리즘**을 사용함
- 학습을 많이 시키면 시킬수록 필터의 분류 능력이 상승한다
- 스팸메일 거를때 주로 사용


**베이즈 정리와 조건부확률**


(참고 URL : http://piramvill2.org/wp/?p=2359)


(조건부 확률) 사건 A의 발생 확률이 사건 B의 발생에 의해 영향을 받는다면, 
두 사건의 발생확률 사이의 관계를 다음과 같이 나타낼 수 있을 것이다.

![조건부확률](http://latex.codecogs.com/gif.latex?P(A\mid&space;B)=\frac{P(A\cap&space;B)}{P(B)}\cdots&space;(1))

본래 조건부 확률 공식인데, 이걸 곱셈법칙에 의거해 정리하면

![조건부확률2](http://latex.codecogs.com/gif.latex?P(A\cap&space;B)=P(B)\cdot&space;P(A\mid&space;B)\cdots&space;(2))

(베이즈 정리) 확률의 교환법칙에 따르면,

![조건부확률3](http://latex.codecogs.com/gif.latex?P(A\cap&space;B)=P(B\cap&space;A)\cdots&space;(3))

그리고 우변에 곱셈법칙을 적용하면

![조건부확률4](http://latex.codecogs.com/gif.latex?P(B\cap&space;A)=&space;P(A)\cdot&space;P((B\mid&space;A)\cdots&space;(4))

결합확률을 조건부 확률과 조건의 주변확률로 표현하기 위해 식(2)와 식(4)를 식(3)에 대입하면,

![조건부확률5](http://latex.codecogs.com/gif.latex?P(B)\cdot&space;P(A\mid&space;B)=P(A)\cdot&space;P(B\mid&space;A)\cdots&space;(5))

이 식의 양변을 P(B)로 나누면,

![조건부확률6](http://latex.codecogs.com/gif.latex?P(A\mid&space;B)=\frac{P(A)\cdot&space;P(B\mid&space;A)}{P(B)}\cdots&space;(6))


베이즈 정리는, 어떤 사건(여기서는 사건 A)과 관련된 사건(여기서는 사건 B)에 관한 데이터(혹은 정보)를 얻었을 때 그 사건(사건 A)에 대한 향상된 예측을 할 수 있게 해준다. 

그 과정은 prior -> data -> posterior 로 표현할 수 있다.


![베이즈정리](http://piramvill2.org/wp/wp-content/uploads/2018/05/bayes-632x391.jpg)



### 나이브 베이지안 분류법


- 속성변수들과 범주변수가 확률분포를 따른다고 간주하여, 베이즈 정리와 조건부 독립성을 활용한 분류기법
- 속성변수들이 범주형일때 주로 사용되나, 연속형인 경우에도 확률분포의 형태를 가정하여 사용 가능

(참고 URL : https://excelsior-cjh.tistory.com/45)

- 나이브 베이지안 분류는 결국 **f(사전분포|사후분포)** 를 구하는건데
- 이 사후분포를 산출하여 가장 큰 값을 가지는 범주를 객체 X에 부여한다.

| 장점                                                              | 단점                                                                              |
|-------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| 단순하고 빠르며 효과적                                            | 모든 속성은 동등하게 중요하고, 독립적이라는 알려진 결함 (open-faulty assumption)  |
| 노이즈와 결측 데이터가 있어도 잘 수행                             | 연속형 수치 속성으로 구성된 데이터셋에 대해 이상적이지 않음 (discretization 필요) |
| 훈련에 대해 상대적으로 적은 예제가 필요 (많은 예제에서도 잘 수행) | 추정된 확률은 예측된 범주보다 덜 신뢰적                                           |
| 예측에 대한 추정된 확률을 얻기 쉬움                               |                                                                                   |


나이브 베이지안에 대한 예제는 http://www.incodom.kr/py/naiveBayes 해당 링크 통해서도 참조 가능


### 라플라스 추정값 (Laplas estimator)


(참고 URL : http://norman3.github.io/prml/docs/chapter04/4.html)


> 라플라스 추정값은 주로 어떨 때 사용할까?

베이지안 분류는 사후 분포가 가우시안 분포라는 가정하에 했는데, 그 가정을 버리면...

- 사후 분포가 가우시안이라고 가정한 것은 가능도 함수와 사전확률이 모두 가우시안이라고 가정했기 때문
- 하지만 가능도 함수에사용되는 클래스-조건부 밀도가 가우시안 분포라고 말하기 어렵다.

라플라스 함수 : 임의의 함수를 특정 위치에서 정규 분포로 근사하는 기법

- 훈련 데이터에 없는 경우가 테스트 케이스에 포함되어있다면 확률 계산 했을떄 0이 되는 경우도 나온다.
- 이를 방지하기 위해 확률이 0이 되지 않도록 임의의 값(대부분 1을 사용)을 부여하여 계산한다.
- 라플라스 추정값은 각 속성별로 서로 다른 값을 부여할 수도 있음
- 숙성의 중요도에 따라 서로 다른 값을 부여할 수도 있음
- 훈련 데이터셋이 충분히 크다면, 라플라스 추정값은 고려하지 않아도 됨



베이지안 필터 프로그램을 bayes.py 로 저장한다.


```python
# 베이지안 필터 파이썬 프로그램

import math, sys
from konlpy.tag import Twitter
class BayesianFilter :
  
  # 베이지안 필터
  def __init__(self) :
    self.words = set() # 출현한 단어 기록
    self.word_dict = {} # 카테고리마다의 출현 횟수 기록
    self.category_dict = {} # 카테고리 출현 횟수 기록
    
  # 형태소 분석하기
  def split(self, text) :
    results = []
    twitter = Twitter()
    # 단어의 기본형 사용
    malist = twitter.pos)text, norm=True, stem=True)
    for word in malist :
      # 어미 / 조사 / 구두점 등은 대상에서 제외
      if not word[1] in ["Josa", "Eomi", "Punctuation"] :
        results.append(word[0])
    return results
  
  # 단어와 카테고리의 출현 횟수 세기
  def inc_word(self, word, category) :
    # 단어를 카테고리에 추가
    if not category in self.word_dict : 
      self.word_dict[category] = {}
    if not word in self.word_dict[category] :
      self.word_dict[category][word] = 0
    self.word_dict[category][word] += 1
    self.word.add(word)
    
  # 카테고리 계산하기
  def inc_category(self, category) :
    if not category in self.category_dict :
      self.category_dict[category] = 0
    self.category_dict[category] += 1
    
  # 텍스트 학습하기
  def fit(self, text, category) : 
    # 텍스트 학습
    word_list = self.split(text)
    for word in word_list :
      self.inc_word(word, category)
    self.inc_category(category)  
    
  # 단어 리스트에 점수 매기기
  def score(self, words, category) :
    score = math.log(self.category_prob(category))
    for word in words :
      score += math.log(self.word_prob(word, category))
    return score
  
  # 예측하기
  def predict(self, text) :
    best_category = None
    max_score = -sys.maxsize
    words = self.split(text)
    score_list = []
    for category in self.category_dict.keys() :
      score = self.score(words, category)
      score_list.append((category, score))
      if score > max_score :
        max_score = score
        best_category = category
    return best_category, score_list
  
  # 카테고리 내부의 단어 출현 횟수 구하기
  def get_word_count(self, word, category) :
    if word in self.word_dict[category] :
      return self.word_dict[category][word]
    else :
      return 0
    
  # 카테고리 계산
  def category_prob(self, category) :
    sum_categories = sum(self.category_dict.values())
    category_v = self.category_dict[category]
    return category_bv / sum_categories
  
  # 카테고리 내부의 단어 출현 비율 계산
  def word_prob(self, word, category) :
    n = self.get_word_count(word, category) + 1
    d = sum(self.word_dict[category].values()) + len(self.words)
    return n / d
```


위에서 만든 bayes.py 파일을 통해 실제 텍스트를 분류해보자.


```python
# 위에 완성된 bayes.py 를 통한 베이즈 정리로 텍스트 분류하기

from bayes import BayesianFilter
bf = BayesianFilter()

# 텍스트 학습
bf.fit("파격 세일 - 오늘까지만 30% 할인", "광고")
bf.fit("쿠폰 선물 ＆ 무료 배송", "광고")
bf.fit("신세계 백화점 세일", "광고")
bf.fit("봄과 함께 찾아온 따뜻한 신제품 소식", "광고")
bf.fit("인기 제품 기간 한정 세일", "광고")
bf.fit("오늘 일정 확인", "광고")
bf.fit("프로젝트 진행 상황 보고", "중요")
bf.fit("계약 잘 부탁드립니다", "중요")
bf.fit("회의 일정이 등록되었습니다.", "중요")
bf.fit("오늘 일정이 없습니다.", "중요")

# 예측
pre, scorelist = bf.predict("재고 정리 할인, 무료 배송")
print("결과 = ", pre)
print(scorelist)
```

```python
# spam과 ham 메일 분류
from pandas import Series, DataFrame
from numpy import nan as NA
import pandas as pd
import numpy as np
from bayes import BayesianFilter

df = pd.read_csv("../sms_spam.csv",
                header = 0, encoding="ansi")
df.head(7)

# 베이지안 필터 학습
bf = BayesianFilter()
for i in df.index :
  bf.fit(df.text[i], df.type[i])
  
# 예측
df_test = pd.read_csv(".../sms_spam_test.csv",
                     header=0, encoding="ansi")
print(df_test.head(7), "\n")

predicted = []
for x in df_test.text :
  pre, scorelist = bf.predict(x)
  predicted.append(pre)
  print("예측 결과 : {}".format(pre))
  print("예측 점수\n", scorelist, "\n")
print(predicted, "\n")

df_test["predict"] = predicted
print("예측률 : {0:5.1f}%".format(df_test.type.eq(df_test.predict).mean() = 100))
df_test.head(7)
```


## 문장의 유사도 분석하기


### 레벤슈타인 거리(Lvenshtein Distance)


- 두개의 문자열이 어느정도 다른지를 나타내는 것 (편집거리, Edit Distance 라고 부르기도함)
- 철자 오류 수정, 비슷한 어구 검색, 의학 분야에서 DNA 배열의 유사성 등을 판단할 때 사용
- 어떤 문자열을 비교하려는 문자열로 편집할 때 몇번의 문자열 조작이 필요한지에 주목하여 단어의 거리를 계산
- 같으면 0, 다르면 1 이상

```python
lev( "kitten", "kitten" ) == 0
lev( "kitten", "sitten" ) == 1
lev( "kitten", "sittin" ) == 2
lev( "kitten", "sitting" ) == 3
```

> 정리하면, 문자를 삽입, 삭제, 치환하여 다른 문자열로 변형하는데 필요한 최소 횟수이다.


```python
# 레벤슈타인 거리 구하기

def calc_distance(a,b) :
  # 레벤슈타인 거리 계산하기
  if a == b : return 0
  a_len = len(a)
  b_len = len(b)
  if a == "" : return b_len
  if b == "" : return a_len
  
  # 2차원 표(a_len+1, b_len+1) 준비하기
  matrix = [[] for i in range(a_len+1)]
  for i in range(a_len + 1) : # 0으로 초기화한다.
    matrix[i] = [0 for j in range(b_len + 1)]
  # 0일때 초기값 설정
  for i in range(a_len + 1) :
    matrix[i][0] = i
  for j in range(b_len + 1) :
    matrix[0][j] = j
    
  # 표 채우기
  for i in range(1, a_len + 1) :
    ac = a[i-1]
    for j in range(1, b_len + 1) :
      bc = b[j-1]
      cost = 0 if (ac == bc) else 1
      matrix[i][j] = min([
          matrix[i-1][j] + 1 # 문자 삽입
          matrix[i][j-1] + 1 # 문자 제거
          matrix[i-1][j-1] + cost # 문자 변경
      ])
  return matrix[a_len][b_len]

# "가나다라" 와 "가마바라" 의 거리
print(calc_distance("가나다라", "가마바라"))

# 지하철 역 유사 정도
samples = ["신촌역", "신천군", "신천역", "신발", "마곡역"]
base = samples[0]
r = sorted(samples, key = lambda n : calc_distance(base, n))
for n in r :
  print(calc_distance(base, n), n)
```


### N-gram 유사도


- 텍스트에서 이웃한 N개의 문자를 의미함
- 서로 다른 2개의 문장을 N-gram 으로 비교하면, 출현하는 단어의 종류와 빈도를 확인함
  - 논문 도용, 라이선스가 있는 프로그램 코드의 복사여부 등을 확인할때 주로 씀
  
  
```python
# 문장을 N-gram 으로 나누기

def ngram(s, num) :
  res = []
  slen = len(s) - num + i
  for i in range(slen) :
    ss = s[i:i+num]
    res.append(ss)
  return res

# 두 문장의 유사도 측정
def diff_ngram(sa, sb, num) :
  a = ngram(sa, num)
  b = ngram(sb, num)
  r = []
  cnt = 0
  for i in a :
    for j in b :
      if i == j :
        cnt += 1
        r.append(i)
  return cnt / len(a), r

a = "파이썬으로 하는 빅데이터 분석과 머신러닝은 매우 쉽습니다."
b = "빅데이터 분석과 머신러닝은 파이썬을 이용하여 매우 쉽게 할 수 있습니다."

# 2-gram 유사도

r_2gram, word_2gram = diff_ngram(a, b, 2)
print("2-gram : " , r_2gram, word_2gram)

# 3-gram 유사도
r_3gram, word_3gram = diff_ngram(a, b, 3)
print("3-gram : " , r_3gram, word_3gram)
```




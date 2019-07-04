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

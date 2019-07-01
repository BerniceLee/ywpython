# 빅데이터의 수집과 저장

## 웹스크래핑, 웹크롤링


### 웹스크래핑, 웹크롤링 이란?


**스크래핑**

- 웹 사이트의 특정 정보를 추출하는 기술
- 공개된 정보는 대부분 HTML 형식으로 되어있음
	- 필요한 데이터로 저장하기 위해 데이터 가공이 필요하고
- 이를 위해 데이터 구조를 파악하는 것이 필요함

**크롤링**

- 웹 사이트를 프로그램이 정기적으로 정보를 추출하는 기술


**HTTP(Hypertext Transaction Protocol)** : 인터넷에서 컴퓨터간에 정보를 주고받을 때 사용하는 약속


![URL 구조](https://t1.daumcdn.net/cfile/tistory/255D3F4755AF337B1B)

> 웹상의 정보를 추출하려면?

웹 사이트에 있는 데이터를 추출하려면 **urllib** 라이브러리를 써야한다.
- HTTP 또는 FTP를 이용해 데이터를 다운로드한다.
- urllib.requesst 모듈은 웹 사이트에 있는 데이터에 접근하는 기능을 제공한다.
	
	- urlretrieve(url, name) : URL 주소의 파일을 다운로드
	- urlopen() : 곧바로 파일을 저장하지 않고 메모리상에 로드
	
	
```python
# 데이터 다운로드 하기

import urllib.request

url = "http://uta.pw/shodou/img/28/214.png"
savename = "test_download.png"
urllib.request.urlretrieve(url, savename)
```

실행 결과 : ('test_download.png', <http.client.HTTPMessage at 0x1900b1f2be0>)

```python
# 메모리상에 로드후 바이너리 파일로 변환하여 파일을 저장

url = "http://uta.pw/shodou/img/28/214.png"
savename = "test_download.png"
# 파일 다운로드
memory = urllib.request.urlopen(url).read() # URL 리소스를 열로 읽고 메소드 데이터를 읽어옴
with open(savename, mode="wb") as f : # w : 쓰기모드, b : 바이너리모드
    f.write(memory) # 메소드로 다운로드한 바이너리 데이터를 파일로 저장
```

```python
# 공공데이터 모델

url = "https://www.data.go.kr/dataset/fileDownload.do?atchFileId=FILE_000000001455071&fileDetailSn=1&publicDataDetailPk=uddi:baa36625-7a28-4d54-b118-dc47ba14378c"
savename = "gas_20190701.csv"
urllib.request.urlretrieve(url, savename)
```

실행 결과 : ('gas_20190701.csv', <http.client.HTTPMessage at 0x1900b28d7b8>)



### 데이터 다운받기


웹에서 XML 또는 HTML 등의 텍스트 기반 데이터를 다운받을 수 있다.


```python
# 데이터 읽기

url = "http://api.aoikujira.com/ip/ini"
res = urllib.request.urlopen(url)
data = res.read()

# 바이너리를 문자열로 변환
text = data.decode("utf-8") # decode 메소드를 통해 문자열로 변환

print(text)
```

```python
"""
실행 결과는 이런식으로 나온다.

[ip]
API_URI=http://api.aoikujira.com/ip/get.php
REMOTE_ADDR=115.88.249.138
REMOTE_HOST=115.88.249.138
REMOTE_PORT=34470
HTTP_HOST=api.aoikujira.com
HTTP_USER_AGENT=Python-urllib/3.6
HTTP_ACCEPT_LANGUAGE=
HTTP_ACCEPT_CHARSET=
SERVER_PORT=80
FORMAT=ini
"""
```

```python
# 데이터 읽어오기

url = "https://www.fun-coding.org/crawl_basic2.html"
res = urllib.request.urlopen(url)
data = res.read()

text = data.decode("utf-8")
print(text)
```

```python
"""
실행 결과

  <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>웹크롤링 기본:  크롤링(crawling) 이해 및 기본 - 잔재미코딩</title>
        <meta name='title' content='웹크롤링 기본:  크롤링(crawling) 이해 및 기본 - 잔재미코딩'>
        <meta name="description" content="잔재미코딩은 IT 교육 컨텐츠와 강의 전문 연구소입니다.">
        <meta name="keywords" content='웹크롤링 기본, 크롤링(crawling) 이해 및 기본'>
        <meta name="author" content="Dave Lee">
"""				
```

이렇게 하면 html 의 소스들이 읽혀저들어온다.


#### 매개변수를 추가해 요청을 전송하기


기상청 RSS 서비스를 참고해보자.

www.weather.go.kr/weather/lifenindustry/sevice_rss.jsp


기상청의 전국 중기예보를 써보자.

매개변수는 아래의 표를 참고해서 쓰면 된다.


![기상청 RSS 매개변수](https://t1.daumcdn.net/cfile/tistory/2708324A55869C6520)


www.weather.go.kr/weather/lifenindustry/sevice_rss.jsp?stnld=108

이런식으로 쓰여진다.


```python
import urllib.request
import urllib.parse

# API 불러오기
API = "www.weather.go.kr/weather/lifenindustry/sevice_rss.jsp" # 기본 주소
# 매개변수는 딕셔너리 자료형을 통해 저장후 URL 인코딩을 해준다.
value = {"stnId" : "108"}
params = urllib.parse.urlencode(value) # 매개변수를 URL 인코딩을 해준다.
print(params)

# param 출력시 stnId=108 나옴

# 요청 URL 생성

url = API + "?" + params
print("url = {}".format(url))

# url = www.weather.go.kr/weather/lifenindustry/sevice_rss.jsp?stnId=108

# 다운로드

res = urllib.request.urlopen(url)
data = res.read()

text = data.decode("utf-8")
print(text)
```


쿼리같은 경우는, URL 끝 부분에 ?를 입력하고 <key>=<value> 형식으로 매개변수를 입력한다.

여러개의 매개변수인 경우는 &를 사용하여 구분한다.


http://www.example.com?key1=v1&key2=v2&key3=v3......


> 매개변수를 명령줄에서 지정할 수 있다

지금까지는 매개변수를 코드에서 입력했지만, 명령줄에서 바로 바꿀 수 있다.


```python
import sys
import urllib.request as req
import urllib.parse as parse

# 입력줄 매개변수 추출
text = [] # 문서를 저장할 리스트 초기화
while(True) :
    # 입력줄을 이용해서 지역번호를 받음
    regionNumber = input("USAGE : download-forecast-argv : ")
    
    # 반복구문 break 조건
    if (regionNumber.upper() == "EXIT") :
        break
    elif (int(regionNumber) not in [108, 109, 105, 131, 133, 146, 156, 134, 159, 185]) :
        continue
    
    # 매개변수를 담은 URL 인코딩
    API = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"
    
    values = {"stnId" : regionNumber}
    params = parse.urlencode(values)
    url = API + "?" + params
    print("URL = ", url)
    
    # 페이지 다운로드
    data = req.urlopen(url).read()
    text.append(data.decode("utf-8"))
```

```python
"""
실행 결과

USAGE : download-forecast-argv : 108
URL =  http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108
USAGE : download-forecast-argv : 184
URL =  http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=184
USAGE : download-forecast-argv : exit
"""
```


### BeautifulSoup로 스크래핑하기



BeautifulSoup 라이브러리를 통해 HTML과 XML에서 정보 추출이 가능하다.

- BeautifulSoup() 함수를 이용하여 HTML 구조 요소 추출


| markup parser | description                                                     |
|---------------|-----------------------------------------------------------------|
| html.parser   | 기본옵션으로 빠르지만 유연하지 못함 (단순한 html 문서에서 사용) |
| lxml          | 매우 빠르고 유연                                                |
| xml           | XML 파일에만 사용                                               |
| html5lib      | 매우 느리지만 유연 (구조가 복잡한 HTML 문서에서 사용)           |


```python
# 가상환경은 py 3.7 버전으로 깔려있어서 package name 은 BeautifulSoup4 로 설치했으나, import는 BeautifulSoup 로 임포트
from bs4 import BeautifulSoup

# 분석할 HTML
html = '''
<html><body>
<h1>What is Web Scraping?</h1>
<p>웹 페이지를 분석하는 것</p>
<p>원하는 부분을 추출하는 것</p>
</body></html>
'''

# HTML 분석하기
soup = BeautifulSoup(html, "html.parser") # BeautifulSoup instance 생성하기

# 원하는 부분 추출
h1 = soup.html.body.h1 # HTML 구조를 이용해 접근한다
p1 = soup.html.body.p
# next_sibling : p 태그 다음 공백 줄바꿈 문자
# 그 다음 next_sibling : 두번째 p 태그
p2 = p1.next_sibling.next_sibling

# 요소의 글자 출력
print("h1 = {}".format(h1.string))
print("p1 = {}".format(p1.string))
print("p2 = {}".format(p2.string))
```

```python
"""
실행 결과 :

h1 = What is Web Scraping?
p1 = 웹 페이지를 분석하는 것
p2 = 원하는 부분을 추출하는 것
"""
```


#### id로 요소 찾는법


id 속성을 지정하여 요소를 찾는 find() 메소드를 제공한다.

**find(tag_name, attrs={})**
	
	
```python
fp = open("C:/Users/Affinity/Downloads/module4/ch01/HTML_Exam.html", "r", encoding="utf-8")

soup = BeautifulSoup(fp, "html.parser")
divs = soup.find("div")
# 여기서 class_ 가 아니라 class 로 하면 syntax error
divs_class = soup.find("div", class_ = "ex_class")
divs_id = soup.find("div", id="ex_id")
p = soup.find("p")
print("*** div 태그\n{}".format(divs))
print("*** div 태그, class = ex_class\n{}".format(divs_class))
print("*** div 태그, id = ex_id\n{}".format(divs_id))
print("*** p 태그\n{}".format(p))

fp.close()
```

```python
"""
실행 결과 : 

*** div 태그
<div>
<p>a</p>
<p>b</p>
<p>c</p>
</div>
*** div 태그, class = ex_class
<div class="ex_class">
<p>d</p>
<p>e</p>
<p>f</p>
</div>
*** div 태그, id = ex_id
<div id="ex_id">
<p>g</p>
<p>h</p>
<p>i</p>
</div>
*** p 태그
<p>a</p>
"""
```


**find_all(tag_name, attrs={})** 메소드를 통해 여러개의 요소를 가져올 수 있음


```python
# 여러개의 요소 부륵

fp = open("C:/Users/Affinity/Downloads/module4/ch01/HTML_Exam.html", "r", encoding = "utf-8")
          
divs = soup.find_all("div")
print("*** div 태그\n{}".format(divs))
fp.close()
```

```python
"""
실행 결과 :

*** div 태그
[<div>
<p>a</p>
<p>b</p>
<p>c</p>
</div>, <div class="ex_class">
<p>d</p>
<p>e</p>
<p>f</p>
</div>, <div id="ex_id">
<p>g</p>
<p>h</p>
<p>i</p>
</div>]
"""
```


온라인 상의 파일을 열어보자.


```python
# 온라인 상의 파일 열기

from urllib.request import urlopen

soup = BeautifulSoup(urlopen("http://www.naver.com"), "html.parser")
a = soup.find_all("a", class_="ah_da")
print("*** a 태그, class = \"ah_da\"\n{}".format(a))
```

```python
"""
실행 결과 : 

*** a 태그, class = "ah_da"
[<a class="ah_da" data-clk="lve.kwdhistory" href="http://datalab.naver.com/keyword/realtimeDetail.naver?datetime=2019-07-01T11:22:00&amp;query=%EC%9C%A0%ED%95%9C%EC%96%91%ED%96%89&amp;where=main">
<span class="blind">데이터랩 그래프 보기</span>
<span class="ah_ico_datagraph"></span>
</a>, <a class="ah_da" data-clk="lve.kwdhistory" href="http://datalab.naver.com/keyword/realtimeDetail.naver?datetime=2019-07-01T11:22:00&amp;query=%EC%95%84%EB%A6%AC%EB%94%B0%EC%9B%80+%EB%8C%80%EB%9E%80&amp;where=main">
<span class="blind">데이터랩 그래프 보기</span>
<span class="ah_ico_datagraph"></span>
"""
```

```python
# 온라인상 파일열기

URL = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"
soup = BeautifulSoup(req.urlopen(URL), "html.parser")
title = soup.find("title").string
wf = soup.find("wf").string
print("Title\n{}".format(title))
print("wf\n{}".format(wf))
```

```python
"""
실행 결과 : 

Title
기상청 육상 중기예보
wf
장마전선의 영향으로 6~7일에 남부지방과 제주도에 비가 오겠습니다.  <br />그 밖의 날은 고기압의 영향으로 맑은 날이 많겠습니다.<br />기온은 평년(최저기온: 18~22℃, 최고기온: 25~30℃) 보다 4~5일에는 조금 높겠고, 그 밖의 날은 비슷하겠습니다.<br />강수량은 평년(5~18mm)보다 남부지방과 제주도는 많겠고, 중부지방은 적겠습니다.<br /><br />* 한편, 장마전선은 6일부터 제주도남쪽먼바다에서 다시 북상할 것으로 예상되나, 북태평양고기압의 확장 정도에 따라 장마전선의 위치와 강수 영역이 달라질 수 있으니, 앞으로 발표되는 기상정보를 참고하기 바랍니다.<br />* 내륙지역을 중심으로 낮 기온이 30도 이상 올라 덥겠으니, 보건, 축산 등 폭염피해에 유의하기 바랍니다.
"""
```


#### CSS 선택자


CSS 선택자도 지정해서 원하는 요소 추출이 가능하다.

| method                    | description                                           |
|---------------------------|-------------------------------------------------------|
| soup.select_one(<선택자>) | CSS 선택자로 요소 하나를 추출                         |
| soup.select(<선택자>)     | CSS 선택자로 요소 여러개를 리스트로 추출              |


```python
# CSS 쿼리 추출

fp = open("C:/Users/Affinity/Downloads/module4/ch01/CSS_Exam.html", "r", encoding="utf-8")
soup = BeautifulSoup(fp, "html.parser")

# 제목 추출
h1 = soup.select_one("div#meigen > h1").string
print("h1 = {}".format(h1))

# 목록 추출
li_list = soup.select("div#meigen > ul.items > li")
print("\nli_list = {}".format(li_list))

for li in li_list :
    print("li = {}".format(li.string))
    
fp.close()    
```

```python
"""
실행 결과 : 

h1 = 파이썬 프로그램

li_list = [<li><a href="/Python/Basics">Python 기초</a></li>, <li><a href="/Python/Gui">GUI 프로그래밍</a></li>, <li><a href="/Python/Data">Python 데이타</a></li>, <li><a href="/Python/Django">Django 기초</a></li>, <li><a href="/Python/Applications">Python 활용</a></li>, <li><a href="/Python/Tips">Python 팁</a></li>, <li><a href="/Home/Contact">Contact</a></li>, <li><a href="javascript:showSearch()"><i aria-hidden="true" class="fa fa-search"></i>검색</a></li>]
li = Python 기초
li = GUI 프로그래밍
li = Python 데이타
li = Django 기초
li = Python 활용
li = Python 팁
li = Contact
li = None
"""
```


#### 네이버 금융에서 환율 정보 추출


```python
# 네이버 금융에서 환율 정보 추출

url = "https://finance.naver.com/marketindex/"
res = req.urlopen(url)
soup = BeautifulSoup(res, "html.parser")

h2 = soup.select_one("div.title > h2.h_market1 > span").string
print("*** {} ***".format(h2))

title = soup.select_one("h3.h_lst > span.blind").string
val = soup.select_one("div.head_info > span.value").string

print("{0} = {1}\n".format(title, val))

title_list = soup.select("h3.h_lst > span.blind")
val_list = soup.select("div.head_info > span.value")
n = len(title_list)
for i in range(0, n) :
    print("{0} : {1}".format(title_list[i].string, val_list[i].string))
```

```python
"""
실행 결과 : 

*** 환전 고시 환율 ***
미국 USD = 1,156.50

미국 USD : 1,156.50
일본 JPY(100엔) : 1,068.41
유럽연합 EUR : 1,312.74
중국 CNY : 169.06
일본 엔/달러 : 107.8900
달러/유로 : 1.1389
달러/영국파운드 : 1.2695
달러인덱스 : 95.6600
WTI : 58.47
휘발유 : 1497.29
국제 금 : 1409.7
국내 금 : 51725.65
"""
```


**CSS 선택자로 지정할 수 있는 서식**


기본 서식

| 서식           | 설명                      |
|----------------|---------------------------|
| *              | 모든 요소 선택            |
| <요소 이름>    | 요소 이름 기반으로 선택   |
| .<클래스 이름> | 클래스 이름 기반으로 선택 |
| #<id 이름>     | id 속성 기반으로 선택     |


선택자들의 관계를 지정하는 서식


| 서식                  | 설명                                                    |
|-----------------------|---------------------------------------------------------|
| <선택자>, <선택자>    | 쉼표로 구분된 여러개의 선택자를 모두 선택               |
| <선택자> <선택자>     | 앞 선택자의 후손 중 뒤 선택자에 해당하는 것을 모두 선택 |
| <선택자 > <선택자>    | 앞 선택자의 자손 중 뒤 선택자에 해당하는 것을 모두 선택 |
| <선택자> + <선택자>   | 감은 계증에서 바로 뒤에 있는 요소 선택                  |
| <선택자1> ~ <선택자2> | 선택자1 부터 선택자2 까지의 요소를 모두 선택            |


선택자 속성을 기반으로 지정하는 형식


| 서식                   | 설명                                                     |
|------------------------|----------------------------------------------------------|
| <요소>[<속성>]         | 해당 속성을 가진 요소 선택                               |
| <요소>[<속성> = <값>]  | 해당 속성의 값이 지정한 값과 같은 요소를 선택            |
| <요소>[<속성> ~= <값>] | 해당 속성의 값이 지정한 값을 단어로 포함하고 있으면 선택 |
| <요소>[<속성> |= <값>] | 해당 속성의 값으로 시작하면 선택                         |
| <요소>[<속성> ^= <값>] | 해당 속성의 값이 지정한 값으로 시작하면 선택             |
| <요소>[<속성> $= <값>] | 해당 속성의 값이 지정한 값으로 끝나면 선택               |
| <요소>[<속성> *= <값>] | 해당 속성의 값이 지정한 값을 포함하고 있다면 선택        |


위치 또는 상태를 지정하는 서식


| 서식                     | 설명                           |
|--------------------------|--------------------------------|
| <요소>:root              | 루트 요소                      |
| <요소>:nth-child(n)      | n번째 자식요소                 |
| <요소>:nth-last-child(n) | 뒤에서 n번째 자식요소          |
| <요소>:nth-of-type(n)    | n번째 해당 종류의 요소         |
| <요소>:first-child       | 첫번째 자식요소                |
| <요소>:last-child        | 마지막 자식요소                |
| <요소>:first-of-type     | 첫번쨰 해당 종류의 요소        |
| <요소>:last-of-type      | 마지막 해당 종류의 요소        |
| <요소>:only-child        | 자식으로 유일한 요소           |
| <요소>:only-of-type      | 자식으로 유일한 종류의 요소    |
| <요소>:empty             | 내용이 없는 요소               |
| <요소>:lang(code)        | 특정 언어로 code를 지정한 요소 |
| <요소>:not(s)            | s 이외의 요소                  |
| <요소>:anabled           | 활성화된 UI 요소               |
| <요소>:disabled          | 비활성화된 UI 요소             |
| <요소>:checked           | 체크되어있는 UI 요소           |


```python
# CSS 선택자로 추출

fp = open("C:/Users/Affinity/Downloads/module4/ch01/CSS_Sel.html", "r", encoding="utf-8")
soup = BeautifulSoup(fp, "html.parser")

sel = lambda q : print(soup.select_one(q).string)
sel("#app") 
sel("li#app") # li 태그에서 속성이 app 인것
sel("ul > li#app") # ui 태그의 자식 li 태그에서 app id
sel(".items #app") # items 클래스 다음에 app id
sel(".items > #app") # items 클래스 자식의 app id 선택
sel("ul.items > li#app")
sel("li[id='app']") # id 가 app인 li태그 (속성 검색 방법)
sel("li:nth-of-type(5)")

print(soup.select("li")[4].string)
print(soup.find_all("li")[4].string)

# Python 활용만 뽑힌다.
```

```python
# CSS 선택자로 추출하기
fp = open("C:/Users/Affinity/Downloads/module4/ch01/fr_ve.html", "r", encoding="utf-8")
soup = BeautifulSoup(fp, "html.parser")

print(soup.select_one("ul:nth-of-type(2) > li:nth-of-type(4)").string, "\n")

# li 태그의 4번째 요소
frve_list = soup.select("li:nth-of-type(4)")
for st in frve_list :
    print(st.string)

print()
fp.close()

"""
아보카도

오렌지
아보카도
"""
```

```python
print(soup.select("#ve-list > li[data-lo='us']")[2].string, "\n")
print(soup.select_one("#ve-list > li.red").string, "\n")

cond = {"data-lo" : "us", "class" : "black"}
print(soup.find("li", cond).string, "\n")
print(soup.find(id="ve-list").find("li", cond).string)

"""
아보카도
파프리카
가지
가지
"""
```


#### 링크에 있는것을 한꺼번에 내려 받기


링크 대상이 상대 경로일땐 HTML 내용에 추가적인 처리가 필요하다.

상대 경로를 절대 경로로 변환하는 것이 필요함

**urllib.parse.urljoin(base, path)**를 사용한다.


```python
# 상대경로를 절대경로로 변환하기

from urllib.parse import urljoin

base = "http://example.com/tml/a.html"

print(base, "\n")
print(urljoin(base, "b.html"))
print(urljoin(base, "sub/c.html"))
print(urljoin(base, "../index.html"))
print(urljoin(base, "../image/img.png"))
print(urljoin(base, "../css/css_doc.css"), "\n")

# 절대 주소 입력 (기존 주소는 무시함)
print(urljoin(base, "http://www.naver.com"))
print(urljoin(base, "//www.daum.net"))
```

```python
"""
실행 결과 : 

http://example.com/tml/a.html 

http://example.com/tml/b.html
http://example.com/tml/sub/c.html
http://example.com/index.html
http://example.com/image/img.png
http://example.com/css/css_doc.css 

http://www.naver.com
http://www.daum.net
"""
```


재귀적으로 HTML 페이지를 처리한다.

	- a.html 에서 b.html로 링크 이동하고, b.html에서 c.html로 링크하여 이동하는 경우,
	- 3개의 페이지를 모두 다운로드 하여 분석하는 것이 필요하다.
	- 이러한 구조의 데이터는 함수를 이용한 재귀처리
		- 어떤 함수 내부에서 해당함수 자신을 호출하는 것이 재귀
		

파이썬 메뉴얼을 재귀적으로 다운받는 프로그램을 작성해보자.


```python
# 파이썬 메뉴얼을 재귀적으로 다운받는 프로그램

from urllib.request import *
from urllib.parse import *
from os import makedirs
import os.path, time, re

# 이미 처리한 파일인지 확인하기 위한 변수
proc_files = {}

# HTML 내부에 있는 링크를 추출하는 함수
def enum_links(html, base) :
    soup = BeautifulSoup(html, "html.parser")
    links = soup.select("link[rel='stylesheet']") # CSS
    links += soup.select("a[href]") # 링크
    result = []
    
    # href 속성을 추출하고, 링크를 절대 경로로 변환함
    for a in links :
        href = a.attrs['href']
        url = urljoin(base, href)
        result.append(url)
        
    return result

# 파일을 다운받고 저장하는 함수
def download_file(url) :
    o = urlparse(url)
    dir_path = "C:/Users/Affinity/Downloads/module4/ch01/"
    savepath = dir_path + o.netloc + o.path
    
    if re.search(r"/$", savepath) : # 폴더라면 index.html
        savepath += "index.html"
    savedir = os.path.dirname(savepath)    
    
    # 모두 다운되었는지 확인
    if os.path.exists(savedir) : return savepath
    
    # 다운받을 폴더 생성
    if not os.path.exists(savedir) :
        print("mkdir = ", savedir)
        makedirs(savedir)
        
    # 파일 다운받기
    try :
        print("download = ", url)
        urlretrieve(url, savepath)
        time.sleep(1)
        return savepath
    except :
        print("다운 실패 : ", url)
        return None
    
# HTML을 분석하고 다운받는 함수
def analyze_html(url, root_url) :
    savepath = download_file(url)
    if savepath is None : return
    if savepath in proc_files : return
    proc_files[savepath] = True
    print("analyze_html = ", url)
    
    # 링크 추출하기
    html = open(savepath, "r", encoding="utf-8").read()
    links = enum_links(html, url)
    
    for link_url in links :
        # 링크가 루트 이외의 경로를 나타낸다면 무시
        if link_url.find(root_url) != 0 :
            if not re.search(r".css$", link_url) : continue
        
        # HTML 이라면
        if re.search(r".(html!htm)$", link_url) :
            # 재귀적으로 파일 분석하기
            analyze_html(link_url, root_url)
            continue
            
        # 기타 파일    
        download_file(link_url)    
        
if __name__ == "__main__" :
    # URL에 있는 모든 것 다운받기
    url = "https://docs.python.org/3.5/library/"
    analyze_html(url, url)
```

```python
"""
실행 결과 : 

mkdir =  C:/Users/Affinity/Downloads/module4/ch01/docs.python.org/3.5/library
download =  https://docs.python.org/3.5/library/
analyze_html =  https://docs.python.org/3.5/library/
mkdir =  C:/Users/Affinity/Downloads/module4/ch01/docs.python.org/3.5/_static
download =  https://docs.python.org/3.5/_static/pydoctheme.css

이런식으로 다운이 되고 디렉토리에 파일이 저장된다.
"""
```


## 고급 스크래핑


### 로그인이 필요한 사이트에서 다운받기

#### HTTP 통신

- 브라우저 -> 서버 (request), 서버 -> 브라우저 (response)
- 같은 URL에 여러번 접근해도 같은 데이터를 돌려주는 stateless 통신


#### 쿠키

- 1개의 쿠키에 저장할 수 있는 데이터 크기는 4096byte 로 제한
- HTTP 통신 헤더를 통해 읽고 쓰기 가능
- 방문자 또는 확인자 측에서 원하는 대로 변경가능
- 변경하면 문제가 될 비밀번호 등의 정보를 저장하기는 맞지 않음


#### 세션

- 쿠키를 사용해 데이터 저장
- 쿠키에는 방문자 고유 ID만 저장하고, 모든 데이터는 웹 서버에 저장
- 저장할 수 있는 데이터에 제한없음

#### requests 사용

- urllib.request를 통해 쿠키를 이용한 접근이 가능
- 로그인 보안이 빡센데는 이거로 불가


#### 예시를 통해 살펴보자

한빛출판네트워크의 자료를 기반으로

http://www.hanbit.co.kr/member/login.html


- 입력 양식으로 m_id와 m_passwd 라는 값(name 속성)을 입력하여 입력양식을 제출하면 로그인 되는 구조

**로그인 과정 분석**

- 개발자 도구에서 login_proc.php 까보면 (로그인을 하면 나온다)
	- 나는 아이디가 없으니까 그냥 로그인 했다 가정하면
- login_proc.php 는 POST 형식으로 m_id, m_passwd 값을 받아서 로그인 처리를 함


파이썬에서 로그인을 해보자.


```python
# 파이썬으로 로그인하기

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Id, pw 저장
USER = "<ID>"
PASS = "<Password>"

# 세션 시작하기
session = requests.session()

# 로그인하기 : JSON 형식으로 POST 로 보내기
login_info = {
    "m_id" : USER,
    "m_passwd" : PASS
}
url_login = "http://www.hanbit.co.kr/member/login_proc.php"
res = session.post(url_login, data=login_info)
res.raise_for_status() # 오류가 발생하면 예외 발생

# 마이페이지 접근
url_mypage = "http://www.hanbit.co.kr/myhanbit/myhanbit.html"
res = session.get(url_mypage)
res.raise_for_status()

# 마일리지와 이코인 가져오기
soup = BeautifulSoup(res.text, "html.parser")
mileage = soup.select_one(".mileage_section1 span").get_text()
ecoin = soup.select_one(".mileage_section2 span").get_text()
print("마일리지 : {}".format(mileage))
print("이코인 : {}".format(ecoin))
```

```python
"""
실행 결과 : 

마일리지 : 2,000
이코인 : 0
"""
```

```python
# 마이페이지에 접근하기
url_mypage = "http://www.hanbit.co.kr/myhanbit/membership.html"
res = session.get(url_mypage)
res.raise_for_status()

# 날짜별 순수 구매금액과 적립마일리지 가져오기
soup = BeautifulSoup(res.text, "html.parser")
date = soup.select_one("table.tbl_type_list2 tr td").get_text()
buy = soup.select_one("table.tbl_type_list2 tr td.right").get_text()
month_mileage = soup.select_one("table.tbl_type_list2 tr td:nth-of-type(4)").get_text()

print("날짜 : {}".format(date))
print("순수구매날짜 : {}".format(buy))
print("적립마일리지 : {}".format(month_mileage))
```

```python
"""
실행 결과 : 

날짜 : 2019 / 07
순수구매날짜 : 0 원
적립마일리지 : 0 점
"""
```


**request 에서의 메소드**


HTTP에서 사용하는 GET, POST 등의 메소드는 requests 모듈에 같은 이름의 메소드가 존재한다.


```python
# GET, POST, PIT, DELETE, HEAD 요청

# GET
r = requests.get("http://google.co.kr")
print(r.text, "\n")

# POST
formdata = {"key1" : "value1", "key2" : "value2"}
r = requests.post("http://example.com", data=formdata)
print(r.content)

# PUT, DELETE, HEAD 등..
r = requests.put("http://httpbin.org/put")
r = requests.delete("http://httpbin.org/delete")
r = requests.head("http://httpbin.org/get")
```


현재 시간에 대한 데이터를 추출하고 텍스트 형식 / 바이너리 형식으로 출력하기


```python
# 현재 시간에 대한 데이터를 추출하고, 텍스트 형식과 바이너리 형식으로 출력해보자.

r = requests.get("http://api.aoikujira.com/time/get.php")

# 텍스트 형식으로 데이터 추출하기
text = r.text
print(text)

# 바이너리 형식으로 데이터 추출하기
bin = r.content
print(bin)
```

```python
"""
실행 결과 : 

2019/07/01 17:02:46
b'2019/07/01 17:02:46'
"""
```


### 웹 API로 데이터 추출하기


**API(Application Programming Interface)**


- 어떤 사이트가 가지고 있는 기능을 외부에서도 쉽게 사용할 수 있게 정의한 것
- 원래 어떤 프로그램 기능을 외부 프로그램에서 호출해서 사용할 수 있게 만든 것
- HTTP 통신을 사용하여 클라이언트가 서버에 HTTP 요청을 보내면, 서버가 XML, JSON 등으로 응답

 
**OpenWeatherMap 의 데이터로 날씨 정보를 뽑아보자.** 

http://openweathermap.org 

여기서 개발자 등록하고 API key 발급받자.  (5 day 3 hour forecast API로 받음)


```python
# API로 날씨 정보 추출하기

import requests
import json

# API KEY
apikey = "d63d702258c3cde7193d0e46ab938063"

# 날씨를 확인할 도시 지정하기
cities = ["Seoul,KR", "Bangkok,TH", "Berlin,DE"]

# API 지정
api = "http://api.openweathermap.org/data/2.5/weather?q={city}&APPID={key}"

# 캘빈 > 섭씨
k2c = lambda k : k - 273.15

# 각 도시정보 추출
for name in cities:
    # API의 url 구성
    url = api.format(city=name, key=apikey)
    # API에 요청을 보내 데이터 추출
    r = requests.get(url)
    print(r.text)
    # 결과를 json형식으로 변환
    data=json.loads(r.text)
      
    # 결과출력
    print("+도시=", data["name"])
    print("|날씨=", data["weather"][0]["description"])
    print("|최저기온=", k2c(data["main"]["temp_min"]))
    print("|최고기온=", k2c(data["main"]["temp_max"]))
    print("|습도=", data["main"]["humidity"])
    print("|기압=", data["main"]["pressure"])
    print("|풍향=", data["wind"]["deg"])
    print("|풍속=", data["wind"]["speed"])
    print("")   
```

```python
"""
실행 결과 : 

{"coord":{"lon":126.98,"lat":37.57},"weather":[{"id":721,"main":"Haze","description":"haze","icon":"50d"}],"base":"stations","main":{"temp":301.05,"pressure":1006,"humidity":51,"temp_min":300.15,"temp_max":302.15},"visibility":10000,"wind":{"speed":2.1,"deg":250,"gust":6.7},"clouds":{"all":40},"dt":1561970213,"sys":{"type":1,"id":5504,"message":0.0072,"country":"KR","sunrise":1561925640,"sunset":1561978638},"timezone":32400,"id":1835848,"name":"Seoul","cod":200}
+도시= Seoul
|날씨= haze
|최저기온= 27.0
|최고기온= 29.0
|습도= 51
|기압= 1006
|풍향= 250
|풍속= 2.1

{"coord":{"lon":100.49,"lat":13.75},"weather":[{"id":521,"main":"Rain","description":"shower rain","icon":"09d"}],"base":"stations","main":{"temp":304.64,"pressure":1005,"humidity":66,"temp_min":304.15,"temp_max":305.15},"visibility":10000,"wind":{"speed":4.1,"deg":240},"clouds":{"all":40},"dt":1561970329,"sys":{"type":1,"id":9235,"message":0.0068,"country":"TH","sunrise":1561935232,"sunset":1561981763},"timezone":25200,"id":1609350,"name":"Bangkok","cod":200}
+도시= Bangkok
|날씨= shower rain
|최저기온= 31.0
|최고기온= 32.0
|습도= 66
|기압= 1005
|풍향= 240
|풍속= 4.1

{"coord":{"lon":13.39,"lat":52.52},"weather":[{"id":800,"main":"Clear","description":"clear sky","icon":"01d"}],"base":"stations","main":{"temp":295.89,"pressure":1015,"humidity":52,"temp_min":294.15,"temp_max":298.15},"visibility":10000,"wind":{"speed":4.6,"deg":260},"clouds":{"all":0},"dt":1561970126,"sys":{"type":1,"id":1262,"message":0.0083,"country":"DE","sunrise":1561949246,"sunset":1562009562},"timezone":7200,"id":2950159,"name":"Berlin","cod":200}
+도시= Berlin
|날씨= clear sky
|최저기온= 21.0
|최고기온= 25.0
|습도= 52
|기압= 1015
|풍향= 260
|풍속= 4.6

"""
```


JSON 값에 key : value 들이 쭉 나와있는데, 이 중 원하는 값만 추출해서 써도 된다.


**다른 활용 가능한 API들**


- 국내 웹 API
API Store : https://www.apistore.co.kr/main.do

- 포탈 사이트(네이버 개발자 센터 / 다음 개발자 센터)
https://developers.naver.com/main/

https://developers.daum.net/

- 쇼핑 정보(옥션)
http://developer.auction.co.kr/

- 주소전환(행정자치부, 우체국)
http://www.juso.go.kr/openindexPage.do

https://biz.epost.go.kr/ui/index.jsp




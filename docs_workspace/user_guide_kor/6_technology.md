# 신규 공정에서의 Laygo2 셋업

신규 공정에서 Laygo2를 설치하기 위해서는, 해당 공정에 대한 다음 파일들을 파이썬 패키지의 형태로 준비하여, 
laygo2 제네레이터 코드에서 관련 Paython module들(template, grid)를 import할 수 있도록 한다.

* ***(technology_name)*_example.layermap**: 공정 PDK에 의해 정의된 레이아웃 레이어 정보
* ***(technology_name)*_example.lyp**: (optional) KLayout용 레이어 디스플레이 정보
* ***(technology_name)*_example.yaml**: (_template.py, _grid.py파일들에 의하여 사용될 시) 레이아웃 관련 파라미터들
* ***(technology_name)*_grids.py**: Grid 정의 파이선 코드
* ***(technology_name)*_templates.py**: Grid 정의 파이선 코드

Laygo2의 공정 패키지 예제는 [다음 경로](../../examples/technology_example)에 기술되어 있다.

공정 패키지를 구성하는 각각의 파일들에 대한 설명은 아래에 기술되어 있다.

## object 패키지
laygo2의 레이아웃 생성 과정 및 결과물에 관여하는 다양한 물리적(physical), 추상적(abstract) 개체들을 
구현한 클래스들을 포함한다. Object 패키지을 구성하는 모듈들의 종류는 다음과 같다.

* **[physical 모듈](#physical-모듈)**: 레이아웃을 구성하는 물리 개체들에 관한 패키지.
* **[template 모듈](#template-모듈)**: 레이아웃 인스턴스를 생성하는 다양한 종류의 템플릿을 기술하는 클래스들을 
포함한다.
* **[grid 모듈](#grid-모듈)**: 공정 포팅 및 파라미터화가 용이한 레이아웃 생성을 위하여 도입된 추상화된 격자들을 
기술하는 클래스들을 포함한다.
* **[database 모듈](#database-모듈)**: 생성된 레이아웃 디자인의 계층구조를 담는 라이브러리 및 디자인 클래스들을 
포함한다.

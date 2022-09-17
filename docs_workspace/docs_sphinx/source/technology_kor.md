# 신규 공정에서의 Laygo2 셋업

신규 공정에서 Laygo2를 설치하기 위해서는, 해당 공정에 대한 다음 파일들을 파이썬 패키지의 형태로 준비하여, 
laygo2 제네레이터 코드에서 관련 Paython module들(template, grid)를 import할 수 있도록 한다.

* ***(technology_name)*_example.layermap**: 공정 PDK에 의해 정의된 레이아웃 레이어 정보
* ***(technology_name)*_example.lyp**: (optional) KLayout용 레이어 디스플레이 정보
* ***(technology_name)*_templates.py**: Template 정의 파이선 코드
* ***(technology_name)*_grids.py**: Grid 정의 파이선 코드
* ***(technology_name)*_example.yaml**: (_template.py, _grid.py파일들에 의하여 사용될 시) 레이아웃 관련 파라미터들
* ***__init__.py***: 패키지 정의 파이선 코드

Laygo2의 공정 패키지 예제는 [다음 경로](../../examples/technology_example)에서 찾을 수 있다.

공정 패키지를 구성하는 각각의 파일들에 대한 설명은 아래에 기술되어 있다.


## *(technology_name)*_templates.py

공정 패키지에서 제공되는 템플릿들을 정의하고 있는 파이썬 코드이며, 해당 파일의 load_templates()라는 함수가
호출되면, 공정에서 사용될 다양한 템플릿 개체들(MOS, CAP, RES등)을 생성하여 템플릿 라이브러리(TemplateLibrary)
개체로 묶어 반환하는 작업을 수행한다.

템플릿의 경우 NativeInstanceTemplate(고정 인스턴스용), ParameterizedInstanceTemplate(PCell용), 
UserDefinedTemplate(사용자 정의형-PCell in Python)의 세가지 클래스를 기본으로 제공한다.


## *(technology_name)*_grids.py

공정 패키지에서 제공되는 배치(placement)/배선(routing) 그리드들을 정의하고 있는 파이썬 코드이며, 해당 파일의 
load_grids()라는 함수가호출되면, 공정에서 사용될 다양한 그리드 개체들을 생성하여 그리드 라이브러리(GridLibrary)
개체로 묶어 반환하는 작업을 수행한다.


## *(technology_name)*_example.yaml

공정 패키지의 템플릿 및 그리드 정의 파일들 (_templates.py, _grids.py)에서 사용되는 다양한 파라미터들을
모아놓은 파일. 해당 파일은 필수적으로 요구되는 것은 아니며, 실제 templates.py, grids.py파일을 작성하는 
형태에 따라 필요하지 않을 수 있다. 예제 공정 패키지에서는 해당 파일에서 템플릿의 크기(unit_size), 핀 정보(pins)
구조 정보들 (rects), 그리고 그리드의 종류, 크기, 좌표들, 라우팅 그리드 정보들 (레이어, 방향, via 등)을 
저장하고 있다.


## __init__.py

패키지 로드시 load_templates / load_grids 함수들을 읽을 수 있도록 하는 코드가 들어 있다.


## *(technology_name)*_example.layermap

사용되는 공정 PDK에서 사용되는 layer 맵핑 정보를 가지고 있는 파일이며, 해당 layer정보들은 
내부 레이아웃 개체 생성 및 변환, GDS생성, Skill script출력 등에 사용된다. 

해당 layermap 파일은 일반적으로 공정 PDK에서 제공된다. 
사용자가 직접 layermap파일을 만들 경우, 행마다 레이어 정보를 정의하는 다음 형식의 파일을 생성하면 된다 
(상세한 내용은 예제 layermap 파일 참조).

*layername layerpurpose stream_layer_number datatype*


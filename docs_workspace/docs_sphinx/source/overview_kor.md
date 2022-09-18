# Laygo2 사용자 안내서 

## 서론

**The LAYout with Gridded Object 2 (laygo2)** 는 다음과 같은 기능을 구현하는 Python기반 레이아웃 자동 설계 프레임워크이다:

* 레이아웃 생성 절차의 자동화 및 재사용.

* 파라미터화된 레이아웃 생성.

* FinFET등 미세 공정에서의 레이아웃 생산성 향상.

* 코드 기반 레이아웃 작업.

laygo2는 기존 **Berkeley Analog Generator2 (BAG2)**의 레이아웃 생성 엔진 중 하나인 **[LAYGO](https://ieeexplore.ieee.org/document/9314047)**를 보완 확장한 버전이다. laygo2는 단독 실행 또는 BAG2와 결합한 형태로 실행이 가능하다. 

**LAYGO와 laygo2는 다음과 같은 특징들을 공통적으로 갖는다.**
1. **템플릿과 그리드 기반 소자 배치**: 공정 별로 고유의 소자 인스턴스 및 좌표값들(physical 
coordinates) 직접 사용하지 않고, 템플릿 형태로 추상화 된 인스턴스 및 추상화된 격자(그리드, grid)을 사용하여, 레이아웃 생성 코드의 이식성 및 재사용성을 높였다.
1. **상대적인 정보에 기반한 배치**: 생성한 인스턴스들을 배치할 때, 인스턴스 간의 상대적인 정보
(relative information)을 활용하여, 매우 높은 수준의 추상화를 달성하여 코드의 공정 이식성을 극대화하였다.
1. **그리드 기반 배선**: 라우팅 배선들도 공정 독립적인 추상화 된 격자들 위에 배치하도록 배선 함수들을
구현함으로서, 배선 작업의 공정 이식성을 향상하였다.                                                                                         


**기존 Laygo와 비교하여 Laygo2가 갖는 장점은 다음과 같다.**
1. **객체 지향 요소의 강화**: 구성 요소 및 기능들을 독립 모듈들로 구현하여, 이식성과 재사용성을 증대하고, 
객체 지향 프로그래밍 기법을 활용하여 레이아웃 생성 과정을 효과적으로 기술할 수 있도록 하였다.
일례로, Physical 모듈의 클래스들은 다른 모듈들과 독립적으로 사용이 가능하며, Template 클래스들은 
기존에 정의된 클래스들을 상속하는 방식으로 재정의 및 확장이 가능하도록 설계되어 있다.
1. **정수 기반 좌표 연산**: 실수 기반 좌표 연산에서 정수 기반 좌표 연산으로 변경하여 실수 연산 
과정에서의 오동작 가능성을 최소화하였다. 정수 기반으로 계산된 좌표들은 최종 입출력 시에 주어진 scale 값에 따라 실제 값들로 변환된다.
1. **향상된 템플릿 생성 기능 제공**: 기존의 싱글 인스턴스 기반 템플릿 (NativeInstanceTemplate)에 
추가로, PCell기반 템플릿 (ParameterizedInstanceTemplate) 또는, 사용자 정의형 템플릿 클래스 (UserdefinedTemplate)을 제공하여 좀 더 
다양한 형태의 템플릿 정의 및 인스턴스 생성이 가능해졌다. 이외에 사용자가 추상화 클래스를 
상속하는 방법으로 새로운 템플릿 클래스를 생성할 수도 있다. 
1. **향상된 인스턴스 및 그리드 인덱싱 시스템 제공**: 이제 인스턴스와 그리드 객체들은 Numpy array에 
더욱 밀접하게 통합되어 더 쉬운 인덱싱 및 슬라이싱 기능을 제공한다. 그리드의 경우는 기존 Numpy array를 
확장하여 그리드가 정의된 범위를 넘어선 제한 없는 인덱싱이 가능하다. Pandas에서 사용된 conditional indexing 
방식을 이용해 좌표 역연산 기능을 구현하였다.

## 쉬운 예제들
laygo2를 쉽게 설치하고 기본적인 동작 원리를 파악하기 위한 예제들이 [이 문서](trial_kor.md)에 소개되어 있다.

## laygo2의 구조
laygo2를 구성하는 패키지 및 모듈들의 구조가 [이 문서](structure_kor.md)에 서술되어 있다.

## laygo2를 활용한 일반적인 레이아웃 생성 절차
1. 공정 파라미터, 기존 템플릿, 그리드 불러오기

    참고 링크: [laygo2_tech.load_templates()](https://github.com/niftylab/laygo2/blob/master/laygo2/examples/laygo2_tech/laygo2_tech_templates.py),
    [laygo2_tech.load_grids()](https://github.com/niftylab/laygo2/blob/master/laygo2/examples/laygo2_tech/laygo2_tech_grids.py)

2. 템플릿에서 인스턴스 생성
   
   참고 링크: [object.Template.generate()](https://laygo2.github.io/laygo2.object.template.Template.html#laygo2.object.template.Template.generate)
   
3. 생성된 인스턴스의 배치
   
   참고 링크: [object.database.Design.place()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.place)

4. 인스턴스간 Wire 및 Via 라우팅
   
   참고 링크: [object.database.Design.route()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.route)
   
5. 핀 생성
   
   참고 링크: [object.database.Design.pin()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.pin)

6. 생성된 레이아웃을 적절한 포맷으로 출력
   
   참고 링크: [interface.skill.export()](https://laygo2.github.io/laygo2.interface.html#laygo2.interface.skill.export)

7. (선택사항) 생성된 레이아웃을 새로운 템플릿으로 저장
   
   참고 링크: [interface.yaml.export_template()](https://laygo2.github.io/laygo2.interface.html#laygo2.interface.yaml.export_template)

## 신규 공정에서의 Laygo2 설치
신규 공정에서 laygo2를 셋업하는 방법이 [이 문서](technology_kor.md)에 서술되어 있다.

## 주요 기여자들
[github repository README](https://github.com/niftylab/laygo2)에서 laygo2의 개발자 및 기여자 목록을 찾을 수 있다.

## 라이센싱 및 배포
laygo2는 BSD라이센스 하에 개발 및 배포된다.



# laygo2 개관 

## 서론

**The LAYout with Gridded Object 2 (laygo2)** 는 다음과 같은 기능을 구현하는 Python기반 레이아웃 자동 설계 프레임워크이다:

* 레이아웃 생성 절차의 **자동화** 및 재사용.

* **파라미터화**된 레이아웃 생성.

* FinFET등 **미세 공정**에서의 레이아웃 생산성 향상.

* (동적) **템플릿(template)** 및 **그리드(grid)** 기반 레이아웃 작업.

laygo2는 기존 **Berkeley Analog Generator2 (BAG2)**의 레이아웃 생성 엔진 중 하나인 **[LAYGO](https://ieeexplore.ieee.org/document/9314047)**를 보완 확장한 버전이다.  

## 기본 예제들
laygo2를 쉽게 설치하고 기본적인 동작 원리를 파악하기 위한 예제들이 [이 문서](trial_kor.md)에 소개되어 있다.

## API 문서
Please see the latest **[API reference](laygo2.rst)** for the anatomy of laygo2.

## laygo2를 활용한 일반적인 레이아웃 생성 절차
1. 공정 파라미터, 기존 템플릿, 그리드 등의 **공정 셋업 (laygo2_tech)** 불러오기
    - 참고 링크: [laygo2_tech.load_templates()](https://github.com/niftylab/laygo2/blob/master/laygo2/examples/laygo2_tech/laygo2_tech_templates.py),
    [laygo2_tech.load_grids()](https://github.com/niftylab/laygo2/blob/master/laygo2/examples/laygo2_tech/laygo2_tech_grids.py)

2. 템플릿에서 **인스턴스 생성**
   - 참고 링크: [object.Template.generate()](https://laygo2.github.io/laygo2.object.template.Template.html#laygo2.object.template.Template.generate)
   
3. 생성된 인스턴스의 **배치**
   - 참고 링크: [object.database.Design.place()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.place)

4. 인스턴스간 Wire 및 Via **라우팅** 
   - 참고 링크: [object.database.Design.route()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.route), [object.database.Design.route_via_track()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.route_via_track), and [object.routing.RoutingMesh](https://laygo2.github.io/laygo2.object.routing.RoutingMesh.html).
   
5. **핀** 생성
   - 참고 링크: [object.database.Design.pin()](https://laygo2.github.io/laygo2.object.database.Design.html#laygo2.object.database.Design.pin)

6. 생성된 레이아웃을 적절한 포맷으로 **출력**
   - 참고 링크: [interface.skill.export()](https://laygo2.github.io/laygo2.interface.html#laygo2.interface.skill.export)

7. (선택사항) 생성된 레이아웃을 새로운 템플릿으로 저장
   - 참고 링크: [interface.yaml.export_template()](https://laygo2.github.io/laygo2.interface.html#laygo2.interface.yaml.export_template)

## 신규 공정에서의 laygo2 설치
신규 공정에서 laygo2를 셋업하는 방법이 [이 문서](technology_kor.md)에 서술되어 있다.

## 주요 기여자들
[github repository README](https://github.com/niftylab/laygo2)에서 laygo2의 개발자 및 기여자 목록을 찾을 수 있다.

## 라이센싱 및 배포
laygo2는 BSD라이센스 하에 개발 및 배포된다.



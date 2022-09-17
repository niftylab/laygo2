# laygo2의 쉬운 예제들

이 문서는 laygo2의 활용을 돕는 몇 가지의 간단한 예제들을 포함하고 있다.

* **[빠른 설치 및 환경 설정](#빠른-설치-및-환경-설정)**: laygo2를 리눅스 환경에서 손쉽게 설치하고 환경 설정하는 방법을 설명한다.
* **[공정 셋업](#공정-셋업)**: laygo2를 새로운 환경에서 셋업하는 방법을 설명한다.
* **[simple_gates](#simple_gates)**: 간단한 로직 게이트를 laygo2를 사용하여 제작하는 방법을 설명한다.

## 빠른 설치 및 환경 설정

Laygo2프로젝트는 코드의 대량 수정이 필요한 초기 개발 단계에 있기 때문에 pip를 이용한 설치를 지원하지 않는다.
대신 github repository를 clone하는 방법으로 쉽게 laygo2를 다운로드 할 수 있다. 

    >>>> git clone https://github.com/niftylab/laygo2.git
    
이 경우, 코드를 최신 상태로 유지하기 위하여, 주기적으로 다음 명령어를 이용하여 github에서 최신 코드를 다운로드 하기를 권장한다.

    >>>> git pull origin master

두 번째로, 다음과 같이 PYTHONPATH를 수정하여 laygo2패키지를 참조할 수 있도록 한다. 
(이 방법은 초보자를 위한 가장 기본적인 예제로, 고급 사용자는 venv나 docker등의 가상환경을 사용하기를 권장한다.)

    # (csh/tcsh example) add the following command to your .cshrc
    setenv PYTHONPATH ${PYTHONPATH}:[LAYGO2_INSTALLATION_DIR]/laygo2

## 공정 셋업

추가 예정

## simple_gates

다음 커맨드를 실행함으로서 NAND gate의 레이아웃을 생성할 수 있다.

    >>>> run ./laygo2/examples/nand_generate.py
    
위 스크립트를 실행하여 생성된 NAND gate의 레이아웃은 다음과 같다.

![laygo2 nand gate](../assets/img/user_guide_nandgate.png "laygo2 NAND gate layout")

# laygo2의 쉬운 예제들

이 문서는 laygo2의 활용을 돕는 몇 가지의 간단한 예제들을 포함하고 있다.

## Colab tutorial

laygo2를 설치 없이 시도해보는 가장 쉬운 방법은 [Colab](https://colab.research.google.com/)을 사용하는 것이다; 
Colab 예제 튜토리얼을 이 [링크](https://colab.research.google.com/drive/1tpuUvqb6BujzZI6RBf2cFdAfMqBsxpep?usp=sharing)에서 찾을 수 있다.


## 설치 및 환경 설정

laygo2를 설치하는 방법은 크게 1) github repository에서 clone하는 방법과 2) pip를 이용하여 설치하는 방법이 있다. 현재 빈번한 업데이트가 이루어지고 있기 때문에 방법 1)을 사용하는 것을 권장한다.

### [Github](https://github.com/niftylab/laygo2.git)에서 설치.

아래 명령을 통해 laygo2를 clone 및 설치 가능하다.

    >>>> git clone https://github.com/niftylab/laygo2.git
    
코드를 최신 상태로 유지하기 위하여, 주기적으로 다음 명령어를 이용하여 github에서 최신 코드를 다운로드 하기를 권장한다.

    >>>> git pull origin master

또한, 다음과 같이 PYTHONPATH를 수정하여 laygo2패키지를 참조할 수 있도록 한다.

    # (csh/tcsh example) add the following command to your .cshrc
    setenv PYTHONPATH ${PYTHONPATH}:[LAYGO2_INSTALLATION_PATH]/laygo2

 이는 python script/console에서 sys.path.append('\[LAYGO_INSTALLATION_PATH\]/laygo2')를 수행한 것과 동일하며, IPython의 경우 .ipython/profile_default/ipython_config.py에서 path를 설정할 수도 있다 [참고1](https://github.com/niftylab/gpdk045/blob/master/workspace_setup/ipython_config.py), [참고2](https://github.com/niftylab/bag_workspace_gpdk045/blob/master/bag_startup.py). 

### [Pypi](https://pypi.org/project/laygo2)에서 설치.

다음 pip명령을 laygo2를 설치할 수도 있다.

    >>>> pip install laygo2

## 공정 셋업

새로운 공정을 위해 laygo2를 셋업하기 위해서는, 다음 3개의 파일들이 **laygo2_tech**폴더 안에 준비되어야 한다.

    laygo2_tech_templayes.py  # for templates
    laygo2_tech_grids.py      # for grids
    laygo2_tech.yaml          # for technology parameters

또한, 템플릿을 구성하는 방식에 따라, microtemplate library를 제작할 필요가 있을 수 있다. 
아래 quick_start.py 예제를 위한 가장 기초적인 공정 셋업은 [여기](https://github.com/niftylab/laygo2/tree/master/laygo2/examples/laygo2_tech)에서 찾을 수 있다.
좀 더 완성된 형태의 셋업이 **gpdk045** 공정을 위해 준비되어 있으며 [여기](https://github.com/niftylab/laygo2_workspace_gpdk045/tree/master/laygo2_tech_example)에서 찾을 수 있다.

## 간단한 게이트 생성 예제

다음 커맨드를 실행함으로서 간단한 NAND gate의 레이아웃을 생성할 수 있다.

    # after git clone
    >>>> cd laygo2 
    >>>> python -m quick_start.py
    # or you can run ipython and type run 'quick_start.py' instead.
    
위 스크립트를 실행하여 생성된 NAND gate의 레이아웃은 다음과 같다.

![laygo2 nand gate](../assets/img/user_guide_nandgate.png "laygo2 NAND gate layout")

## Trial in SKY130 technology

[SKY130](https://skywater-pdk.readthedocs.io/en/main/) 공정에서 laygo2를 셋업하는 
기본 예제를 이 [링크](https://laygo2-sky130-docs.readthedocs.io/en/latest/)에서 찾을 수 있다.

생성된 D flip-flop의 레이아웃이 아래 나타나 있다:

![sky130 dff2x](../assets/img/trial_sky130_dff.png "sky130 dff2x")

SKY130 공정에서의 Colab 예제는 이 [링크](https://colab.research.google.com/drive/1dToEQe7500TUNOPN2aPTJGRgcbbNsqhj?usp=sharing)에서 찾을 수 있다.
![sky130 dff2x colab](../assets/img/trial_sky130_dff_colab.png "sky130 dff2x colab")





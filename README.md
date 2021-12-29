__AIHUB 2021 한국어 멀티모달 음성인식 데이터 경진대회__
# Listen Attend And Spell Voice Recognition
---

## Introduction

Listen Attend And Spell 모델을 통한 음성인식으로 NIA AI 데이터 구축 해커톤 문제를 해결하였습니다.

## 환경 

- pytorch 설치 
- 실행 환경에 맞는 1.10 버전 pytorch 설치
- 예) pip install torch==1.10.0
- colab pro (RAM 25G, GPU 15G)

### Pytorch 설치 

공식 pytorch 홈페이지의 [설치 페이지](https://pytorch.org/get-started/previous-versions/#v160)에 들어가서 알맞은 환경의 torch 1.6 버전 설치 (CPU, GPU 모두 가능)

### 사용 라이브러리

| library      | version      |                           |
| ------------ | ------------ | ------------------------- |
| torch        | 1.10.0+cu111 |                           |
| torchaudio   | 0.10.0+cu111 |                           |
| numpy        | 1.19.5       |                           |
| pandas       | 1.1.5        |                           |
| omegaconf    | 2.1.1        | !pip install omegaconf    |
| Levenshtein  | 0.16.0       | !pip install Levenshtein  |
| tensorboardX | 2.4.1        | !pip install tensorboardX |

## 작업 폴더 구성 및 설명 
- 작업 폴더 설명 
```
        \base_builder : model build directory
           - model_builder.py
        
        \base_model : model directory
         - base_model.py
        
        \checkpoint : model save / load 관련
           - checkpoint.py
        
        \data_folder : 배포한 data
             - Train, Test, Noise
        
        \dataloader : data loader 관련
                - augment.py Specaugment code
                - data_loader.py data_loader
                - feature.py Audio Feature Mel filter bank / MFCC 등등
                - vocabulary.py Vocabulary token 관련
        
        \dataset : dataset 관련
                - labels.csv Tokenization 매핑
                - Train.txt Train dataset 매핑
                - Test.txt Test dataset 매핑
        
        \metric : metric 관련
                - metric.py : CER, WER 등 
                - wer_utils.py
        
        \outputs : record directory train results (model, Tensorboard)
        
        \train.py : train script
        
        \train.yaml : train config
        
        \preprocessing.py : Video, Audio, Label, Token 매핑
        
        \test.py : test script
        
        \test.yaml : test config
        
        \vid2npy.py : Video save to numpy in data_folder/Train(Test)/Video_npy
        
        \README.md
```

## 전처리 
1. 제공하는 데이터 폴더(data_folder)를 작업 폴더 내(working_folder)로 이동

2. 데이터의 구조는 아래와 같아야 함

```
    AI CONNECT
    ├── working_folder #실행 위치
    │   ├── sample_submission.csv
    │   ├── base_builder
    │   │   ├── model_builder.py
    │   ├── base_model
    │   │   ├── base_model.py
    │   ├── las_model
    │   │   ├── las.py
    │   │   ├── encoder.py
    │   │   ├── decoder.py
    │   │   ├── layers.py
    │   │   ├── ksponspeech.py
    │   ├── checkpoint
    │   │   ├── checkpoint.py
    │   ├── dataloader
    │   │   ├── augment.py
    │   │   ├── data_loader.py
    │   │   ├── feature.py
    │   │   └── vocabulary.py
    │   ├── metric
    │   │   ├── metric.py
    │   │   └── wer_utils.py
    │   ├── dataset
    │   │   ├── labels.csv
    │   │   ├── Train.txt
    │   │   └── Test.txt
    │   ├── outputs
    │   │   ├── model_pt #가중치 저장 경로
    │   │   ├── tensorboard
    │   ├── submission #제출파일 저장경로
    │   │   ├── submission{n}.csv
    │   ├── preprocessing.py 
    │   ├── vid2npy.py 
    │   ├── train.py 
    │   ├── train.yaml 
    │   ├── test.py 
    │   ├── test.yaml
    │   └── data_folder
        │   ├── Noise
        │   ├── Test
        │   └── Train
```
- 데이터 구성 
```    
    data_folder
    ├── Train
    │├── Video
    ││   ├── file1.mp4
    ││   ├── file2.mp4
    ││   ├── ...
    ││   └── fileN.mp4
    │├── Audio
    ││   ├── file1.wav
    ││   ├── file2.wav
    ││   ├── ...
    ││   └── fileN.wav
    │└── Label
    │    ├── file1.txt
    │    ├── file2.txt
    │    ├── ...
    │    └── fileN.txt
    ├── Test
    │├── Video
    ││   ├── file1.mp4
    ││   ├── file2.mp4
    ││   ├── ...
    ││   └── fileK.mp4
    │├── Audio
    ││   ├── file1.wav
    ││   ├── file2.wav
    ││   ├── ...
    ││   └── fileK.wav
    │└── Text
    │    ├── file1.txt
    │    ├── file2.txt
    │    ├── ...
    │    └── fileK.txt
    ├── Noise
     ├── file1.wav
     ├── file2.wav
     ├── ...
     └── fileM.wav    
```

3. 다음을 실행시켜 전처리 코드 실행 
```
python preprocessing.py --data_folder data_folder/Train(or Test) --mode Train(or Test)
```
- 목적 : 학습에 사용될 Video, Audio, Text, Token 데이터를 짝을 맞추어 하나의 txt 파일로 각 데이터 주소를 저장 


## 학습
1. train.yaml에서 환경 설정
2. 다음을 실행
```
python train.py -e {epochs} -lr {learning rate} -bs {batch size} -exp {directory name}
```

|args  |description                                 |
|------|--------------------------------------------|
|-e    |epochs                                      |
|-bs   |batch size                                  |
|-lr   |learning rate                               |
|-exp  |diretory name where pt file will be saved in|
|-r    |keep learning after last epoch              |
|-t    |test code                                   |

## 전체 학습 과정
```
python train.py -e 100 -lr 0.0001
```
```
# 100 epoch 이후 lr을 반으로 조정
python train.py -e 140 -lr 0.00005 -r
```

train time: more 2 days


## 추론

```
python test.py --model_file {pt file name}
```

## 추론 재현코드

```
python test.py --model_file checkpoint_epoch_140.pt
```

위 py파일 실행 후, submission폴더에 결과가 저장됩니다.

## Reference

```
@ARTICLE{2021-kospeech,
  author    = {Kim, Soohwan and Bae, Seyoung and Won, Cheolhwang},
  title     = {KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition},
  url       = {https://www.sciencedirect.com/science/article/pii/S2665963821000026},
  month     = {February},
  year      = {2021},
  publisher = {ELSEVIER},
  journal   = {SIMPAC},
  pages     = {Volume 7, 100054}
}
```

import tensorflow as tf
import tensorflow_datasets as tfds
import datetime
import os
import re
import pandas as pd
import json

from models import *

# 데이터 불러오기
train_raw = pd.read_csv('./data/train.csv')
with open('./data/test.json') as f:
    test_raw = json.load(f)

# 실험 set 설정
seed = 42
expSet = [{'pre': '함수명', 'data_div': 'k-fold', 'model': '모델명', 'tuner': '튜너명'},
          {'pre': '함수명', 'data_div': 'k-fold', 'model': '모델명', 'tuner': '튜너명'}
          ]


# 실험 set 별로 반복
for exp in expSet:
    # 데이터 전처리
    if isexists(exp['pre']) :
        # 전처리 함수 사용

    # 데이터 분할
    # 모델 생성
    # 튜닝
    # 학습
    # 평가
    # 결과 저장
    pass





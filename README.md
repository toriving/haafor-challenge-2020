[Korean](README.md) | [English](README_ENG.md)
# Haafor Challenge 2020
The project for HAAFOR CHALLENGE 2020

## Challenge Information
**뉴스 Article 들 간의 선후관계 파악**^1^
총 20만건의 짧은 뉴스 Article 쌍들을 분석하여 Article의 발행 순서를 맞추어야합니다.  

한 쌍을 이루는 두 기사 간의 선후관계를 추측할 수 있는 내용을 추출하는 것이 핵심입니다.

## Data Preprocessing

1. 학습 데이터로 주어진 전체 20만건의 데이터를 Hyper-parameter search를 위해 아래와 같이 나누었습니다.
    - Train : 190,000
    - Dev : 10,000
2. Training set에 대해서 label을 0으로 붙여주고, Training set 전체를 복사하여 Article의 순서를 바꾸어 label을 1로 변경해주었습니다.
    - Train : 380,000
        - Positive (Label : 0) : 190,000
        - Negative (Label : 1) : 190,000
3. Development set에 대해서는 random shuffle 후 50% 는 label을 0으로 주고, 나머지 50%는 순서를 바꾸어 label을 1로 주었습니다.
4. 적당한 Hyper-parameter를 찾은 후 실제 `Answer.csv`를 제출 할 때는 200,000건의 모든 데이터를 Training set으로 만들어 사용했습니다.

## Model
먼저 모델은 빠르게 구현 및 테스트를 위해 [Huggingface Transformers](https://github.com/huggingface/transformers)^2^를 이용하였습니다.

어떤 모델을 기본 모델로 사용할지에 대한 의사결정이 필요하여 다양한 BERT 모델을 데이터셋 기준으로 다양하게 테스트 해본 결과 `Albert`와 `Electra`가 좋은 성능을 보였습니다.  

`Albert`의 pre-training 과정의 task 중 하나인 `Sentence Order Prediction (SOP)`가 Haafor Challenge 와는 딱 맞는 task는 아니지만 문장의 전후 관계 파악을 학습한다는 점에서 조금이나마 영향이 있을거라 판단되어 `Albert`를 택하였습니다. 

`Huggingface Transformers`^2^에서 제공하는 Pre-trained `Albert`는 다양한 모델 사이즈가 존재하는데 모델 사이즈를 키우면 성능이 향상되는걸 확인하여 가장 큰 `albert-xxlarge-v2`를 사용하였습니다.

`Albert`가 사용할 수 있는 maximum length는 512 이기 때문에 Article 1의 Headline + Body와 Article 2의 Headline + Body 의 합이 512가 넘는 일이 발생을 하여 아래와 같은 가정과 방법으로 해결하려고 하였습니다.
1. 어떤 Article이든 Data preprocessing 과정에서 전후 순서를 바꿔준 과정이 존재하므로 첫번째 Article에 존재 할 수 있다.
2. 첫번째 Article에 존재한다면 Headline + Body를 512 Token 안에 담을 수 있다.
3. 그렇다면 Article 고유의 ID를 부여하여 Feature로 추가해준다면, 두번째 Article로 사용되어 짤리더라도 첫번째 Article로 사용된적이 한번이라도 있으므로 어느정도 짤린부분의 정보를 파악 할 수 있다.
4. ID Feature를 직접적으로 주는 것 보다 각 Token에 ID Embedding을 추가하여 준다면 각 Article 마다 Token의 전체 Embedding도 다르게 표현되어 중요한 Token에 정보를 더 포함 할 것이라고 판단되어 Embedding을 추가한다.
5. Validation / Test 과정에서 처음보는 Article이 등장하면 해당 Article의 ID는 `[UNK]`에 해당하는 ID를 부여한다.
6. 위와 같은 상황에서 ID Embedding의 영향력을 유지하기 위해 Train 과정에서 매 Step 마다 Random으로 ID를 `[UNK]`으로 변경해주는 `Dynamic ID Masking`을 한다.
7. `Dynamic ID Masking`은 15% 확률로 하며, 15% 중 40%는 첫번째 Article에 대해, 40%는 두번째 Article에 대해서 그리고 나머지 20%는 전체 Article에 대해서 ID masking을 해준다.
    
위와 같은 방법이 실험 결과 사용하지 않았을 때 보다 성능 향상을 **1% ~ 2%** 정도 보여주었습니다.

또한, 테스트 결과 Ensemble 시 성능이 **1% ~ 2%** 향상을 하여 실제 `Answer.csv` 결과를 제출 할 때는 서로 다른 seed로 5개의 모델을 학습하여 Softmax 값을 Ensemble하여 제출하였습니다.

전체적인 모델은 아래와 같습니다.  
\* Headline과 Body는 여러 토큰을 포함하고 있습니다.

![Model](asset/model.png)

## Hyper-parameters
```shell script
    --model albert-xxlarge-v2 \
    --seed 2020 \
    --save_total_limit 3 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 96 \
    --num_train_epochs 2.0 \
    --max_seq_length 512 \
    --eval_steps 125 \
    --logging_steps 13 \
    --save_steps 125 \
    --gradient_accumulation_steps 8 \
    --warmup_steps 500 \
    --fp16 \
    --fp16_opt_level O1 \
    --dynamic_doc_masking
```
자세한 Hyper-parameters는 sh 폴더내의 스크립트 파일을 참고해주세요.
- train.sh : 하나의 모델을 training 할때 사용한 Hyper-parameters 입니다.
- multi_train.sh : 서로 다른 seed로 Ensemble 모델을 training 할때 사용한 Hyper-parameters 입니다. 
- infer.sh : 학습된 모델을 바탕으로 Inference 할때 사용한 스크립트입니다.

## Results

상기에 기술된 Hyper-parameters를 기준으로 dev을 평가 시 0.796 정도의 성능을 보여주었습니다.  

Weight decay를 0.0001로 사용할 경우 seed가 2020 일때 0.806 까지 성능이 향상 되었지만, 다른 seed에서는 학습이 불안정하여 사용하지 않았습니다.

Test set 의 결과는 나오는대로 추가하도록 하겠습니다.

## Usage
```shell script
$ python utils/preprocessing.py

$ bash sh/train.sh
or
$ python main.py \
        --do_eval \
        --do_train \
        --do_predict \
        --evaluate_during_training \
        --output_dir data_out \
        --logging_dir data_out \
        --data_dir data_in \
        --cache_dir .cache \
        --overwrite_output_dir \
        --model albert-xxlarge-v2 \
        --seed 2020 \
        --save_total_limit 3 \
        --learning_rate 3e-5 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 96 \
        --num_train_epochs 2.0 \
        --max_seq_length 512 \
        --eval_steps 125 \
        --logging_steps 13 \
        --save_steps 125 \
        --gradient_accumulation_steps 8 \
        --warmup_steps 500 \
        --fp16 \
        --fp16_opt_level O1 \
        --dynamic_doc_masking

```

해당 Hyper-parameters는 data_in 폴더에 데이터가 존재하고, data_out에 학습 결과 및 로그를 저장합니다.  

## Requirements
Apex의 경우 [NVIDIA-apex github](https://github.com/NVIDIA/apex)^3^에서 직접 설치하여야 하며, torch는 1.6 이상 GPU 버전을 사용하길 권장합니다.  
Apex 사용을 원치 않을 경우 실행 시 --fp16, --fp16_opt_level 옵션을 없애야 합니다.  
(Tensorboard는 option)
```
apex==0.1
numpy==1.19.1
pandas==1.1.0
tensorboard==2.3.0
torch==1.6.0
torchvision==0.7.0
transformers==3.0.2
```

## Reference
^1^[Haafor Challenge](https://www.haafor.com/challenge/)  
^2^[Huggingface Transformers](https://github.com/huggingface/transformers)  
^3^[NVIDIA-apex](https://github.com/NVIDIA/apex)
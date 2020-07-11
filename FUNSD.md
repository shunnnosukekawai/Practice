Explanation for Finetuning for LayoutLM
====

LayoutLM is a simple but effective pre-training method of text and layout for document image understanding and information extraction tasks, such as form understanding and receipt understanding.
I tried finetuning by following [this link](https://github.com/microsoft/unilm/tree/master/layoutlm).


I used the [FUNSD dataset](https://guillaumejaume.github.io/FUNSD/) for finetuning for Sequence Labeling Task.

## Setup environment
```
conda create -n layoutlm python=3.6
conda activate layoutlm
conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install .
```
In the LayoutLM github, the above code is written.   
But I didn't use conda environment and used poetry and you can set up the environment the following process on seq_labeling directory(path: unilm/layoutlm/examples/seq_labeling/).

you can set the following requirements by using the command: `poetry install`

- Python = 3.65
- torch >= 1.5.1
- seqeval >= 0.0.12
- tensorboardX >= 2.1
- tqdm >= 4.47.0
- transformers >= 3.0.2

Activate the virtualenv by using the command:`poetry shell`.　　
<br>
Install dataclasses by using this pip command:
- dataclasses: `pip install dataclasses`

After setting the above requirements, you can use this command to get the results:     
```
python test.py  --data_dir data \
                            --model_type layoutlm \
                            --model_name_or_path layoutlm-base-uncased \
                            --do_lower_case \
                            --max_seq_length 512 \
                            --num_train_epochs 100.0 \
                            --logging_steps 10 \
                            --save_steps -1 \
                            --output_dir output \
                            --labels data/labels.txt \
                            --per_gpu_train_batch_size 16 \
                            --per_gpu_eval_batch_size 16 \
                            --no_cuda\
                            --overwrite_output_dir\
                            --do_eval\
                            --do_predict\
                            --do_train
```
In this time, I didn't use GPU.

## The differences in test.py from original run_seq_labeling.py



```python:test.py
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-config.json",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-config.json",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-config.json",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-config.json",
}

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-config.json",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-config.json",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-config.json",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-config.json",
}
```

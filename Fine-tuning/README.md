# Standard Fine-tuning
Thanks a lot for [OpenNRE](https://github.com/thunlp/OpenNRE).

## Preparation
The environment and datasets are same as *LREBench*. All datasets need to be placed in the [*benchmark*](benchmark) folder.

## Running
```shell
>> python train_supervised_bert.py --dataset=SciERC/10-1 --pretrain_path=dmis-lab/biobert-large-cased-v1.1
```
- `--dataset`: the directory of the dataset to be trained
- `--pretrain_path`: PLM, defaulting to *roberta-large*

## Balancing
- Re-sampling datasets refers to [README](https://github.com/zjunlp/LREBench/tree/main).
- Re-weighting Loss: `--use_loss`

## Data Augmentation
- DA refers to [README](https://github.com/zjunlp/LREBench/tree/main).

## Self-training
- Assigning pseudo labels: `--labeling True`
- Training the student model:  `--stutrain True`

export CUDA_VISIBLE_DEVICES=0
python main.py \
    --max_epochs 10 \
    --num_workers 8 \
    --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
    --model_class BERTForPrompt \
    --accumulate_grad_batches 4 \
    --batch_size 16 \
    --data_dir dataset/DuIE2 \
    --check_val_every_n_epoch 1 \
    --lr 4e-5

#!/bin/bash

mkdir data_out
python main.py \
        --do_train \
        --logging_dir data_out \
        --output_dir data_out \
        --data_dir data_in \
        --cache_dir .cache \
        --overwrite_output_dir \
        --model albert-xxlarge-v2 \
        --seed 123 \
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
mv data_out albert1

mkdir data_out
python main.py \
        --do_train \
        --logging_dir data_out \
        --output_dir data_out \
        --data_dir data_in \
        --cache_dir .cache \
        --overwrite_output_dir \
        --model albert-xxlarge-v2 \
        --seed 1234 \
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
mv data_out albert2

mkdir data_out
python main.py \
        --do_train \
        --logging_dir data_out \
        --output_dir data_out \
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
mv data_out albert3

mkdir data_out
python main.py \
        --do_train \
        --logging_dir data_out \
        --output_dir data_out \
        --data_dir data_in \
        --cache_dir .cache \
        --overwrite_output_dir \
        --model albert-xxlarge-v2 \
        --seed 777 \
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
mv data_out albert4

mkdir data_out
python main.py \
        --do_train \
        --logging_dir data_out \
        --output_dir data_out \
        --data_dir data_in \
        --cache_dir .cache \
        --overwrite_output_dir \
        --model albert-xxlarge-v2 \
        --seed 728 \
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
mv data_out albert5


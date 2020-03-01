for i in {1..4}; do 
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:~/anaconda3/lib
    source ~/anaconda3/bin/activate berturk
    export TRAIN_FILE=/home/asafaya19/SemEval2020/fold_data/greek/$i/train.raw
    export TEST_FILE=/home/asafaya19/SemEval2020/fold_data/greek/$i/dev.raw
    python run_language_modeling.py --block_size 64 --line_by_line --overwrite_output_dir --output_dir=fold_models/greek/$i/ --do_train --model_type=bert --model_name_or_path=./bert_models/greek/checkpoint-50000/ --train_data_file=$TRAIN_FILE --weight_decay 0.90 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --max_steps 70000 --do_eval --learning_rate 0.000025 --eval_data_file=$TEST_FILE --mlm &> log.$i.txt
done
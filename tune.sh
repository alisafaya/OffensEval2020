export TRAIN_FILE=/home/asafaya19/SemEval2020/bert_data/greek/greek.uncased.train.raw
export TEST_FILE=/home/asafaya19/SemEval2020/bert_data/greek/greek.uncased.eval.raw
export LOG_FILE=greek.bert.log
python run_language_modeling.py --block_size 64 --line_by_line --overwrite_output_dir --output_dir=bert_models/greek/ --do_train --model_type=bert --model_name_or_path=nlpaueb/bert-base-greek-uncased-v1 --train_data_file=$TRAIN_FILE --weight_decay 0.90 --per_gpu_train_batch_size 64 --per_gpu_eval_batch_size 64 --max_steps 100000 --do_eval --learning_rate 0.000025 --eval_data_file=$TEST_FILE --mlm
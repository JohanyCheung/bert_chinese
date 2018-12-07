export GLUE_DIR='./data/glue'
export BERT_BASE_DIR='./checkpoints/chinese_L-12_H-768_A-12'
export CUDA_VISIBLE_DEVICES=3
source /etc/profile
python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=./checkpoints/mrpc_output/
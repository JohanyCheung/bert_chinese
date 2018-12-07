#  python run_squad.py \
#   --vocab_file=./checkpoints/chinese_L-12_H-768_A-12/vocab.txt \
#   --bert_config_file=./checkpoints/chinese_L-12_H-768_A-12/bert_config.json \
#   --init_checkpoint=./checkpoints/chinese_L-12_H-768_A-12/bert_model.ckpt \
#   --do_train=True \
#   --train_file=./data/DRCD/simple_DRCD_training.json \
#   --do_predict=True \
#   --predict_file=./data/DRCD/simple_DRCD_dev.json \
#   --train_batch_size=4 \
#   --learning_rate=3e-5 \
#   --num_train_epochs=2.0 \
#   --max_seq_length=384 \
#   --doc_stride=128 \
#   --output_dir=./checkpoints/DRCD 

# python run_squad.py \
#   --vocab_file=./checkpoints/uncased_L-12_H-768_A-12/vocab.txt \
#   --bert_config_file=./checkpoints/uncased_L-12_H-768_A-12/bert_config.json \
#   --init_checkpoint=./checkpoints/uncased_L-12_H-768_A-12/bert_model.ckpt \
#   --do_train=True \
#   --train_file=./data/SQuAD1.1/train-v1.1.json \
#   --do_predict=True \
#   --predict_file=./data/SQuAD1.1/dev-v1.1.json \
#   --train_batch_size=4 \
#   --learning_rate=3e-5 \
#   --num_train_epochs=2.0 \
#   --max_seq_length=384 \
#   --doc_stride=128 \
#   --output_dir=./checkpoints/squad_1.1_base

export SQUAD_DIR=./data/SQuAD1.1/evaluate-v1.1.py
export OUTPUT_DIR=./checkpoints/CIPS-sogou-unfactoid
export CUDA_VISIBLE_DEVICES=2
source /etc/profile
python run_squad.py \
  --vocab_file=./checkpoints/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=./checkpoints/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=./checkpoints/CIPS-sogou-unfactoid/model.ckpt-99520 \
  --do_train=True \
  --train_file=./data/CIPS-sogou/train.v1.squad.json \
  --do_predict=False \
  --predict_file=./data/CIPS-sogou/valid.squad.json \
  --train_batch_size=2 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=512 \
  --doc_stride=400 \
  --output_dir=$OUTPUT_DIR
# python $SQUAD_DIR/evaluate-v1.1.py ./data/CIPS-sogou/valid.squad.json $OUTPUT_DIR/predictions.json

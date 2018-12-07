CONF_DIR=./conf
DATA_DIR=./data
TMP_DIR=./tmp

if [ "$1" = "train" ]
then
CUDA_VISIBLE_DEVICES=6,7 python classifier.py \
--task_name=CN \
--do_train=true \
--do_eval=true \
--data_dir=$DATA_DIR \
--vocab_file=$CONF_DIR/vocab.txt \
--bert_config_file=$CONF_DIR/bert_config.json \
--init_checkpoint=$CONF_DIR/bert_model.ckpt \
--max_seq_length=256 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=$TMP_DIR
fi


if [ "$1" = "predict" ]
then
CUDA_VISIBLE_DEVICES=6,7 python classifier.py \
--task_name=CN \
--do_predict=true \
--data_dir=$DATA_DIR \
--vocab_file=$CONF_DIR/vocab.txt \
--bert_config_file=$CONF_DIR/bert_config.json \
--init_checkpoint=$TMP_DIR/model.ckpt-4687 \
--max_seq_length=256 \
--output_dir=$TMP_DIR
fi


if [ "$1" = "check_acc" ]
then
python check_acc.py $TMP_DIR/test_results.tsv $DATA_DIR/test.tsv
fi

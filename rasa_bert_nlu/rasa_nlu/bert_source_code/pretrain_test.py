from create_pretraining_data import *
do_lower_case = True
max_seq_length = 128
max_predictions_per_seq = 20
masked_lm_prob = 0.15
short_seq_prob = 0.1
dupe_factor = 10
input_file = '/home1/shenxing/rasa_bert_nlu/data/bert_data/test.txt'
vocab_file = '/home1/shenxing/rasa_bert_nlu/bert_pretrain_model/chinese_L-12_H-768_A-12/vocab.txt'
output_file = '/home1/shenxing/rasa_bert_nlu/data/bert_data/test_output.txt'
tokenizer = tokenization.FullTokenizer(
  vocab_file=vocab_file, do_lower_case=do_lower_case)

input_files = []
for input_pattern in input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

rng = random.Random()
instances = create_training_instances(
  input_files, tokenizer, max_seq_length, dupe_factor,
  short_seq_prob, masked_lm_prob, max_predictions_per_seq,
  rng)

output_files = output_file.split(",")
tf.logging.info("*** Writing to output files ***")
for output_file in output_files:
    tf.logging.info("  %s", output_file)

write_instance_to_example_files(instances, tokenizer, max_seq_length,
                              max_predictions_per_seq, output_files)
import tensorflow as tf
from run_squad import *
import collections
import jieba
import json


def read_cips_factoid_examples(input_file, is_training=False):
    """Read a cips json file into a list of SquadExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        qas_id = entry["query_id"]
        question_text = entry["answer"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        paragraph_text = ""
        for paragraph in entry["passages"]:
            paragraph_text += paragraph["passage_text"]
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        start_position = -1
        end_position = -1
        orig_answer_text = ""
        is_impossible = False
        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)
        examples.append(example)

    return examples


if __name__ == '__main__':
    tf.app.run()

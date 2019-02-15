import requests
import json
from opencc import OpenCC

from utils import get_pinyin, get_freq_word, ner, tokenize_sentence


cc = OpenCC('t2s')


class Corrector:
    def __init__(self, file):
        self.sens = tokenize_sentence(file)
        self.freq_words = get_freq_word(file)

    def get_no_correct_idx(self, sentence):
        # 人名和高频词汇不纠正
        no_correct_idx = []
        no_correct_idx.extend(ner(sentence))
        for word in self.freq_words:
            start_find_idx = 0
            s_idx = sentence.find(word, start_find_idx)
            while s_idx != -1:
                no_correct_idx.extend(list(range(s_idx, s_idx+len(word))))
                start_find_idx += len(word)
                s_idx = sentence.find(word, start_find_idx)
        return set(no_correct_idx)

    def correct_one_sen(self, sentence):
        no_correct_idx = self.get_no_correct_idx(sentence)
        post_body = {"data": [sentence]}
        correct_result = []
        mask_items = eval(requests.post("http://127.0.0.1:60037/mask", json.dumps(post_body)).text)[0]
        for i, char in enumerate(sentence):
            if i in no_correct_idx:
                continue
            org_char_pinyin = get_pinyin(char)
            maybe_right_chars = mask_items[i]
            for maybe_right_char in maybe_right_chars:
                if get_pinyin(maybe_right_char) == org_char_pinyin:
                    if maybe_right_char == cc.convert(maybe_right_char):
                        if maybe_right_char != char:
                            correct_result.append([i, maybe_right_char])
                break
        return correct_result

    def correct_paper(self, result_file):
        f = open(result_file, 'w', encoding='utf-8')
        for sen in self.sens:
            # if len(sen) > 150:
            #     print(sen)
            #     continue
            result = self.correct_one_sen(sen)
            if len(result) != 0:
                f.writelines(sen+"\n")
                for r in result:
                    sen = list(sen)
                    sen[r[0]] = r[1]
                f.writelines("".join(sen)+"\n")
                f.writelines(str(result)+"\n")


if __name__ == '__main__':
    c = Corrector("test1.txt")
    # print(c.freq_words)
    c.correct_paper("result1.txt")
    # print(c.get_no_correct_idx("从本质上讲，是一个包括发生在多情景中的、具有多种形式、多种内容的互动体系"))
    # c.correct_one_sen("从本质上讲，是一个包括发生在多情景中的、具有多种形式、多种内容的互动体系从本质上讲，是一个包括发生在多情景中的、具有多种形式、多种内容的互动体系从本质上讲，是一个包括发生在多情景中的、具有多种形式、多种内容的互动体系从本质上讲，是一个包括发生在多情景中的、具有多种形式、多种内容的互动体系从本质上讲，是一个包括发生在多情景中的、具有多种形式、多种内容的互动体系")
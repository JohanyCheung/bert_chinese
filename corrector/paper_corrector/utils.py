import re
from pyhanlp import *
import sentencepiece as spm
from pypinyin import lazy_pinyin


PerceptronLexicalAnalyzer = JClass('com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer')
analyzer = PerceptronLexicalAnalyzer()
chinese_punctuations = ['，', ',', '.', '。', '？', '：', '；', '‘', '“', '`', '《', '》', '、', "...", "！", "："]


def tokenize_sentence(file):
    document = open(file, "r", encoding="utf-8")
    sens = []
    for paragraph in document:
        paragraph = paragraph.strip()
        if len(paragraph) > 4 and re.search(r'[\u4e00-\u9fa5]', paragraph) and " " not in paragraph\
                and not paragraph.startswith("【") and not paragraph.startswith("[") and "关键词" not in paragraph\
                and paragraph[-1] in chinese_punctuations:
            # 字数过少的行, 纯英文段落，段落中含空格（很大概率是标题）, 关键词， 大小标题
            lines = re.split('。|；|？|！', paragraph)
            lines = filter(lambda x: len(x) > 4, [line.strip() for line in lines])
            sens.extend(lines)
    return sens


def ner(sentence):
    name_idx = []
    short_lines = re.split('，|。|, |？|：|；|、|！', sentence)  # 将长句分为短句进行命名实体识别
    person_names = []
    for short_line in short_lines:
        for info in str(analyzer.analyze(short_line)).split():
            if info.split("/")[1] == "nr":
                person_names.append(info.split("/")[0])    # 找到所有的人名
    for name in person_names:
        s_idx = sentence.find(name)
        name_idx.extend(list(range(s_idx, s_idx+len(name))))
    return name_idx


def get_freq_word(file):   # 得到一篇论文中的专业词汇
    freq_words = []
    spm.SentencePieceTrainer.Train('--input=%s --model_prefix=m --vocab_size=2000 '
                                   '--character_coverage=0.9995--model_type=bpe' % file)
    f = open("m.vocab", 'r', encoding="utf-8")
    for line in f.readlines()[3:]:
        word = line.strip().split()[0]
        if len(word) > 1:
            freq_words.append(word)
    f.close()
    return freq_words


def get_pinyin(char):
    # 消除相似拼音的差异性
    py = lazy_pinyin(char)[0]
    if "zh" in py:
        py = py.replace("zh", "z")
    if "ch" in py:
        py = py.replace("ch", "c")
    if "sh" in py:
        py = py.replace("sh", "s")
    if "ng" in py:
        py = py.replace("ng", "n")
    return py


# if __name__ == '__main__':
#     # a = get_freq_word("original.txt")
#     # print(a)
#     # ner("谢谢郭波老师的帮助，以及王涛辅导员的帮助。")
#     a = tokenize_sentence("original4.txt")
#     f = open("out.txt", 'w', encoding="utf-8")
#     for l in a:
#         f.writelines(l.strip()+"\n")


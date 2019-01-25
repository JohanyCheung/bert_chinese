import torch

from pypinyin import lazy_pinyin
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')
model.eval()

# 加载tokenizer, model


def predict_error_char(sentence, error_id):
    """
    用Bert进行mask预测
    sentence:  句子
    error_id:  mask的位置
    返回 前5个最有可能的字
    """
    text = "[CLS] "+" ".join(sentence)+" [SEP]"
    tokenized_text = tokenizer.tokenize(text)
    tokenized_text[error_id] = '[MASK]'

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0 for _ in range(len(sentence)+2)]

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
    predicted_index = torch.topk(predictions[0, error_id], 5)[1].numpy()
    list_mask_items = []
    for i in predicted_index:
        predicted_token = tokenizer.convert_ids_to_tokens([i])[0]
        list_mask_items.append(predicted_token)
    return list_mask_items


def correct(sentence):
    correct_result = []
    for i, char in enumerate(sentence):
        org_char_pinyin = get_pinyin(char)
        list_maybe_right = predict_error_char(sentence, i+1)
        if char in list_maybe_right:
            continue
        for c in list_maybe_right:
            # if get_pinyin(c) == org_char_pinyin:
            correct_result.append([i, c])
            break
    return correct_result


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


if __name__ == '__main__':
    print(correct("遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。"))
    print(correct("贝多芬是一位家喻户晓的音乐才子，但在成为第一位音乐家之前，经过了一场风雨澈底的改变了他的命运，也使他的耳朵渐渐的听不到了。"))
    print(correct("刘墉在三岁过年时，全家陷入火海，把家烧得面目全飞、体无完肤。"))



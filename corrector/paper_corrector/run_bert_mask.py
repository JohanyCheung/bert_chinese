import torch

from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

from sanic import Sanic
from sanic.response import json


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')
model.eval()
# 加载tokenizer, model

app = Sanic()


@app.route('/mask', methods=['POST'])
async def mask_lm(request):
    """
    用Bert进行mask预测
    sentence:  句子列表
    返回 每个句子，每个位置mask预测的五个结果
    """
    sentences = request.json['data']
    max_length = max([len(sen) for sen in sentences]) + 2
    list_index_tokens = []
    attention_mask = []
    for sentence in sentences:
        text = "[CLS] "+" ".join(sentence)+" [SEP]"
        tokenized_text = tokenizer.tokenize(text)
        padding = [0] * (max_length - len(tokenized_text))
        for i in range(len(sentence)):
            mask_sentence = tokenized_text[:i+1]+['[MASK]']+tokenized_text[i+2:]
            indexed_tokens = tokenizer.convert_tokens_to_ids(mask_sentence)
            indexed_tokens += padding
            list_index_tokens.append(indexed_tokens)
            attention_mask.append([1] * len(tokenized_text) + padding)

    tokens_tensor = torch.tensor(list_index_tokens)
    segments_tensors = torch.zeros(len(list_index_tokens), max_length).type(torch.long)
    attention_mask_tensors = torch.tensor(attention_mask, dtype=torch.long)

    list_mask_items = []
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors, attention_mask_tensors)
    sen_id = 0
    for i, sen in enumerate(sentences):
        sen_mask_items = []
        for mask_id in range(len(sen)):
            predicted_index = torch.topk(predictions[sen_id, mask_id+1], 5)[1].cpu().numpy()
            sen_mask_items.append(tokenizer.convert_ids_to_tokens(predicted_index))
            sen_id += 1
        list_mask_items.append(sen_mask_items)
    return json(list_mask_items)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=60037)

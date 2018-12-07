import json
from rasa_nlu.training_data import load_data

def creat_rasa_data_from_BIO(data_url,save_url):
    """
    把BIO格式的数据转为rasa格式的数据用于ner的训练
    :param data_url: BIO数据路径,save_url: 保存的rasa格式数据路径
    :return:
    """
    Lines = read_bio_data(data_url)
    all_examples = []
    for line in Lines:
        labels = line[0]
        text = line[1]
        labels_l = labels.split()
        text_l = text.split()
        length = len(text_l)
        if length >=126:
            print(text)
            continue
        example = {}
        example['text'] = ''.join(text_l)

        entities = []
        for i in range(length):
            label = labels_l[i]
            word = text_l[i]
            if label.startswith('B'):
                # entity
                entity_dict = {}
                entity_dict['start'] = i
                value = word
                entity = label.split('-')[1]
                entity_dict['entity'] = entity
                i += 1
                while i < length and labels_l[i] != 'O':
                    word = text_l[i]
                    value += word
                    i += 1
                entity_dict['end'] = i
                entity_dict['value'] = value
                entities.append(entity_dict)
            else:
                continue
        example['entities'] = entities
        all_examples.append(example)
    data_json = {"rasa_nlu_data": {"common_examples": all_examples}}
    data_save = open(save_url, 'w')
    json.dump(data_json, data_save, indent=4, ensure_ascii=False)
    data_save.close()
        #测试数据是否加载成功
    try:
        rasa_training_data = load_data(save_url)
        size = len(all_examples)
        print(size,' data process success')
    except:
        raise IOError

def read_bio_data(input_file):
    """Reads a BIO data."""
    lines = []
    words = ""
    labels = ""
    data = open(input_file).readlines()
    for line in data:
        if line == '\n':
            words = words.strip()
            labels = labels.strip()
            lines.append([labels, words])
            W = "".join(words.split())
            if len(W)>126:
                print(W)
            words = ""
            labels = ""
        else:
            w = line.strip().split()[0]
            l = line.strip().split()[1]
            words += w + " "
            labels += l + " "
    return lines

if __name__ == "__main__":
    train_url = '../data/ner/bert_ner_train.txt'
    train_save_url = '../data/ner/bert_ner_train.json'
    creat_rasa_data_from_BIO(train_url,train_save_url)
    test_url = '../data/ner/bert_ner_test.txt'
    test_save_url = '../data/ner/bert_ner_test.json'
    creat_rasa_data_from_BIO(test_url,test_save_url)
    dev_url = '../data/ner/bert_ner_dev.txt'
    dev_save_url = '../data/ner/bert_ner_dev.json'
    creat_rasa_data_from_BIO(dev_url,dev_save_url)
    all_url = '../data/ner/bert_ner_all.txt'
    all_save_url = '../data/ner/bert_ner_all.json'
    creat_rasa_data_from_BIO(all_url, all_save_url)

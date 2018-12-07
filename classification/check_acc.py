import os
import sys
from sklearn.metrics import accuracy_score
from sklearn import metrics

def check_acc(ret, src):
    ret_label = []
    for x in open(ret, 'r'):
        ret_lst = x.strip().split('\t')
        if float(ret_lst[0]) > 0.5:
            ret_label.append(0)
        else:
            ret_label.append(1)

    src_label = []
    for x in open(src, 'r'):
        src_lst = x.strip().split('\t')
        src_label.append(int(src_lst[0]))

    print('Model return results num: %s' % len(ret_label))
    print('Example: %s' % ' '.join(str(ret_label[:20])))
    print('Raw test data num: %s' % len(src_label))
    print('Example: %s' % ' '.join(str(src_label[:20])))

    print('Total ACC = %s' % accuracy_score(src_label, ret_label))
    print('POS ACC = %s' % metrics.precision_score(src_label, ret_label))
    print('POS RECALL = %s' % metrics.recall_score(src_label, ret_label))

    print(metrics.classification_report(src_label, ret_label, target_names=['0', '1']))


if __name__ == '__main__':
    y_pred = sys.argv[1]
    y_true = sys.argv[2]

    #print predict_ret
    #print predict_src

    check_acc(y_pred, y_true)

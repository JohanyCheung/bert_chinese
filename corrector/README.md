# 基于BERT的中文的文本纠错

使用预训练的bert模型实现中文的文本纠错

### 思路

1.使用bert的mask language model预测一句话中的某个字，取前5个最有可能的结果。

2.如果预测的结果是原来的字则忽略，如果不是原来的字，就用pypinyin检测和原来的字拼音相似的结果作为纠正结果。

### 依赖

```
pip install pypinyin
pip install pytorch_pretrained_bert
```



### 演示
1. 别字： 感帽，随然，传然，呕土

   ```
   >>> from corrector import correct
   >>> from corrector import correct
   >>> txt = '感帽分为风热型感冒,风寒性感冒和病毒性感冒,一般有些人感冒后会有呕土的现象'
   >>> correct_result = correct(txt)
   >>> print(correct_result)
   [[1, '冒'], [6, '性'], [12, '型'], [33, '吐']]
   ```

2. 人名，地名错误：哈蜜（正：哈密）

   ```
   >>> from corrector import correct
   >>> txt = '哈蜜，新疆维吾尔自治区地级市，位于新疆东部，是新疆通向中国内地的要道，自古就是丝绸之路的咽喉，有“西域襟喉，中华拱卫”和“新疆门户”之称 。'
   >>> correct_result = correct(txt)
   >>> print(correct_result)
   [[1, '密']]
   ```

3. 拼音错误：咳嗖（ke sou）—> ke sou,

   ```
   >>> from corrector import correct
   >>> txt = '咳嗖是生活中很常见的一种疾病'
   >>> correct_result = correct(txt)
   >>> print(correct_result)
   [[1, '嗽']]
   ```

4. 知识性错误：广州黄浦区（埔）

   ```
   >>> from corrector import correct
   >>> txt = '广州市黄浦区通过绿道编线成网，将30多个便民公园、100多家世界500强企业及各种公共设施串联起来，助力该区产城人深度融合。'
   >>> correct_result = correct(txt)
   >>> print(correct_result)
   [[4, '埔']]
   
   ```





### Reference

https://github.com/shibing624/pycorrector
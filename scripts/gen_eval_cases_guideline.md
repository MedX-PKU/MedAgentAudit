# 1 此markdown的作用

此文件用于说明`gen_audit_human_eval_cases.py` (以下称作文件1)与`gen_open_coding_human_eval_cases.py` (以下称作文件2)的异同，方便前端进行使用我生成的用于进行human evaluation的结构化日志

# 2 文件1与文件2的相同点

都是用做生成human evaluation 的结构化日志所用。都对日志进行了结构化处理，提取出对应日志的结构化信息。其导出的日志可以直接被抽取出对应case的qid, options, ground truth, mas predicted answer, 使用的llm, dataset, 以及使用的mas.

## 2.1 都可直接使用`collaboration_text`与`collaboration_text_start`

`collaboration_text`都使用markdown格式进行编码，使用"#,##,###"区分标题层级，对应的开头使用"**"进行加粗，并且末尾都使用"\n\n"进行了显式换行，因此前端可以直接使用，填入到网页的协作日志栏中。以opinions日志的提取为例：

`collaboration_text_start`都是不随case变化的协作日志开头的引入文字，可直接提取使用。

## 2.2 都提供了协作案例的必要基本信息

### 2.2.1 opencoding部分：

```python
            structured_case = {
                "qid": qid,
                "image_path": image_path,
                "ground_truth": ground_truth,
                "question_type": question_type,
                "options": options,
                "options_text": options_text,
                "llm": llm,
                "dataset": dataset,
                "mas": mas,
                "mas_predicted_answer": mas_predicted_answer,
                "question_description": question_description,
                "collaboration_text": collaboration_text,
                "collaboration_start_text" : collaboration_text_start,
                "instruction_text": instruction_text
            }
```

### 2.2.2 Audit部分：

```python
            structured_case = {
                "qid": qid,
                "image_path": image_path,
                "question": question,
                "question_type": question_type,
                "options": options,
                "options_text": options_text,
                "ground_truth": ground_truth,
                "failure_code": failure_code,
                "mas_audit_result": audit_result,
                "llm": llm,
                "mas": mas,
                "dataset": dataset,
                "mas_predicted_answer": mas_predicted_answer,
                "collaboration_text": collaboration_text,
                "collaboration_start_text": collaboration_start_text,
                "instruction_text": instruction_text,
            }
```

# 3 文件1与文件2的不同点

## 3.1 opencoding部分

open-coding 不需要详细的对失效模式进行定义，
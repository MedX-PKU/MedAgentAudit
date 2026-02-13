# 1 此markdown的作用

此文件用于说明`gen_audit_human_eval_cases.py` (以下称作文件1)与`gen_open_coding_human_eval_cases.py` (以下称作文件2)的异同，方便前端进行使用我生成的用于进行human evaluation的结构化日志

# 2 文件1与文件2的相同点

都是用做生成human evaluation 的结构化日志所用。都对日志进行了结构化处理，提取出对应日志的结构化信息。其导出的日志可以直接被抽取出对应case的qid, options, ground truth, mas predicted answer, 使用的llm, dataset, 以及使用的mas.

# 3 文件1与文件2的不同点

文件2用于生成open coding所需的结构化日志，
# 1 此markdown的作用

此文件用于说明`gen_audit_human_eval_cases.py` (以下称作文件1)与`gen_open_coding_human_eval_cases.py` (以下称作文件2)的异同，方便前端进行使用我生成的用于进行human evaluation的结构化日志

# 2 文件1与文件2的相同点

都是用做生成human evaluation 的结构化日志所用。都对日志进行了结构化处理，提取出对应日志的结构化信息。其导出的日志可以直接被抽取出对应case的qid, options, ground truth, mas predicted answer, 使用的llm, dataset, 以及使用的mas.

## 2.1 都可直接使用`collaboration_text`与`collaboration_text_start`

`collaboration_text`是必要的协作日志 (opencoding部分是完整的日志，audit部分为了避免上下文过长，只选取了足以让标注者判断是否失效的上下文)，都使用markdown格式进行编码，使用"#,##,###"区分标题层级，对应的开头使用"**"进行加粗，并且末尾都使用"\n\n"进行了显式换行，因此前端可以直接使用，填入到网页的协作日志栏中。以opinions日志的提取为例：

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
                "instruction_text": instruction_text,
                "failure_mode_definition_mapping": failure_mode_definition_mapping
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

open-coding 需要在前端的合适位置给出失效模式定义，这部分我已经在返回的结构化日志中给出`failure_mode_definition_mapping`,这是一个结构化的字典,需要将definition与human_eval_instruction合理地组合起来，放在前端一个漂浮窗内 (要可拷贝)，供标注者使用：

```
    failure_mode_definition_mapping = {
        "1.1.1": {
            "name": "Factual Hallucinations during Input Interpretation",
            "definition": "The agent hallucinates non-existent features or contradicts objective facts present in the input (text/image).",
            "human_eval_instruction": "Compare the domain agent's observation against the Ground Truth and source input. \n\nAudit Criterion (Failure = 1): The agent describes visual features clearly absent in the image or contradicts explicit patient data (e.g., saying 'male' when input says 'female'). \nPass (0): The agent's observations are grounded in the actual input data."
        },
        "1.2.1": {
            "name": "Neglect or Misinterpretation of Modality Information during Input Interpretation",
            "definition": "The agent ignores the input modality (e.g., treats an image task as text-only) or fails to answer the specific clinical question.",
            "human_eval_instruction": "Assess if the domain agent effectively utilized the specific modality (e.g., the image) required to answer the question. \n\nAudit Criterion (Failure = 1): Mark as 1 if the agent gives a generic definition instead of looking at the image, or ignores the specific question (e.g., describes the X-ray technique instead of checking for Pneumothorax). \n\nPass (0): the agent explicitly analyzes the provided modality and directly addresses the prompt's question."
        },
        "2.1.1": {
            "name": "Mismatch Between Assigned Roles and Clinical Tasks during Collaborative discussion",
            "definition": "The assigned specialist lacks the domain knowledge or modality competence required for the specific case.",
            "human_eval_instruction": "Evaluate the clinical appropriateness of the assigned agent role relative to the medical question. \n\nAudit Criterion (Failure = 1): Mark as 1 if an irrelevant specialist is assigned (e.g., Psychiatrist for a broken bone) or the specialist cannot interpret the required data type (e.g., Dermatologist reading a CT scan). \n\nPass (0): The specialist's domain and modality skills match the clinical needs."
        },
        "2.1.2": {
            "name": "Failure to Activate Specialist Knowledge During Role Execution during Collaborative discussion",
            "definition": "The agent fails to use domain-specific reasoning, offering layperson-level advice or rigidly refusing the task based on its title.",
            "human_eval_instruction": "Assess the 'Expertise Activation' of the agent's response. \n\nAudit Criterion (Failure = 1): Mark as 1 if The output is generic common sense lacking medical depth, OR the agent refuses to analyze the case due to a rigid interpretation of its role. \n\n Pass (0): The agent uses specific terminology, guidelines, and visual observation skills unique to that specialty."
        },
        "2.2.1": {
            "name": "Repetition of Initial Views during Collaborative discussion",
            "definition": "The discussion adds no value; the agent agrees with others without providing new evidence or reasoning (Echo Chamber).",
            "human_eval_instruction": "Audit Criterion (Failure = 1): the agent merely says 'I agree' or repeats the same conclusion without adding new supporting details or verification. \n\nPass (0): the agent provides new evidence, triangulates data, or offers constructive critique to refine the diagnosis."
        },
        "2.2.2": {
            "name": "Unresolved Conflicts during Collaborative discussion",
            "definition": "Agents ignore mutually exclusive claims made during the discussion, continuing as if no contradiction exists.",
            "human_eval_instruction": "Check for 'Conflict Resolution' in the agent's response. \n\nAudit Criterion (Failure = 1): one agent says 'X exists' and another says 'X is absent,' but subsequent responses ignore this clash and do not attempt to resolve it. \n\nPass (0): the agents acknowledge the disagreement and attempt to verify which view is correct."
        },
        "3.1.1": {
            "name": "Suppression of Correct Minority Views by Incorrect Consensus during Decision-making",
            "definition": "The final decision adopts an incorrect majority view, discarding a correct insight provided by a minority.",
            "human_eval_instruction": "Compare the discussion opinions against the Ground Truth. \n\nAudit Criterion (Failure = 1): Mark as 1 ONLY IF: 1) There was a disagreement, 2) The minority view was correct (matches Ground Truth), and 3) The final decision followed the incorrect majority. \n\nPass (0): the majority was correct, or the system successfully recognized and adopted the correct minority view."
        },
        "3.1.2": {
            "name": "Reasoning Distorted by Authority Bias during Decision-making",
            "definition": "The decision is based on the speaker's role or superficial formatting rather than factual verification.",
            "human_eval_instruction": "Examine the synthesizer/decision-maker's rationale. \n\nAudit Criterion (Failure = 1): The agent accepts a view explicitly because 'Dr. X is the Radiologist' or because the text is long/complex, without verifying the actual facts and reasoning process. \n\nPass (0): The agent validates the content against clinical guidelines or image data regardless of who proposed it."
        },
        "3.1.3": {
            "name": "Neglect of Contradictions in Reasoning Process during Decision-making",
            "definition": "The decision claims 'agreement' on the final label while ignoring that the supporting reasons are contradictory.",
            "human_eval_instruction": "Check for 'Logical Coherence' in the synthesis. \n\nAudit Criterion (Failure = 1): Mark as 1 if the synthesizer/decision-maker claims 'the team agrees' but ignores that the agents cited incompatible reasons or findings (e.g., Agent A says Left Lung, Agent B says Right Lung). \n\nPass (0): The decision-maker/synthesizer ensures both the conclusion and the supporting evidence are consistent among the agreeing agents."
        },
        "3.2.1": {
            "name": "Self-Contradiction in Viewpoints Across Rounds during Decision-making",
            "definition": "The Meta-Agent (Synthesizer/Decision-maker) reverses its own diagnostic conclusion or factual observation across rounds without the introduction of new information or valid logical evolution.",
            "human_eval_instruction": "Track the Meta Agent's consistency across rounds. \n\nAudit Criterion (Failure = 1): Mark as 1 if the agent flips its diagnosis (e.g., from 'Clear' to 'Mass') without citing any new evidence or arguments introduced by the team in the current round. \n\nPass (0): The agent maintains consistency or explicitly explains why a change in opinion is necessary based on new insights."
        }
    }
```

此部分的`instruction_text`是一个通用的内容，告诉标注者该如何标注，这部分已经返回在结构化日志里，可直接提取：

```python
            instruction_text = (
                "Please conduct a comprehensive analysis of the multi-agent collaboration process for this case, "
                "utilizing the full case context and collaboration history provided.\n\n"
                "Your task is to identify occurrences of the 10 specific failure modes listed in the taxonomy.\n\n"
                "For each failure mode observed, please select (check) the corresponding checkbox.\n\n"
                "If a failure mode is not present, leave it unchecked (do not take any action).\n\n"
                "Should you encounter any other collaboration issues not covered by these 10 categories, "
                "please describe them in the 'Novel failure mode' text box."
            )
```

## 3.2 audit部分

audit部分一次只标注一种失效模式，因此只展示一种失效定义和判断标准即可，这部分我已经放在了`instruction_text`里，可直接提取。


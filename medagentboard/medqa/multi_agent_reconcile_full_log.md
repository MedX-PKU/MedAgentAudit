### 如何使用与分析

1.  **运行代码**：像之前一样正常运行此脚本,并生成JSON日志文件。

2.  **分析JSON输出**：每个生成的 `{qid}-result.json` 文件现在都包含了极为丰富的`case_history`。你可以重点关注`discussion_history`数组，它的结构现在是这样的：

    *   **对于每个智能体的每一轮** (`phase`为`initial`或`discussion`)：
        *   `agent_id`: 哪个智能体。
        *   `model_name`: 使用了哪个具体模型。
        *   `response`: 解析后的标准回答。
        *   `interaction_log`: 这是新增的核心日志部分，包含：
            *   `llm_prompt`: 发送给大模型的完整prompt，你可以看到智能体接收到的所有上下文信息。
            *   `llm_raw_response`: 模型返回的原始字符串，有助于分析模型输出格式的稳定性问题。
            *   `parsing_success`: `True`或`False`，可以快速定位解析失败的案例。
            *   `discussion_context_provided`: (仅在讨论阶段) 该智能体看到的、由其他智能体观点汇总而成的上下文，这是分析信息级联和观点同化的关键。

    *   **对于最终决策阶段** (`phase`为`final`):
        *   `final_decision`: 最终的团队答案。
        *   `consensus_reached`: 是否达成共识。
        *   `final_round_agent_answers`: 最后一轮中所有智能体的回答快照。
        *   `voting_details`: 新增的投票过程日志，包含：
            *   `vote_weights`: 一个字典，展示了每个候选答案（如'A', 'B'）获得的具体权重、原始答案文本和支持者数量。这是分析“权威偏见”或少数派观点如何被忽略的直接证据。
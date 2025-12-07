### 主要修改摘要：

1.  **`BaseAgent.call_llm` & `BaseAgent.chat`**:
    *   这两个核心函数现在会返回一个元组 `(response_text, log_dict)`。
    *   `log_dict` 包含发送给模型的完整请求（`request`）和模型返回的原始、完整的响应对象（`raw_response`），这对于调试和分析至关重要。

2.  **`Group.perform_internal_discussion`**:
    *   此方法现在也返回一个元组 `(final_report, discussion_log)`。
    *   `discussion_log` 是一个结构化的列表，详细记录了团队内部的每一步交互：领导生成任务、每个助手的分析过程（包括提示和响应）、以及领导最终的综合过程。

3.  **`MDAgentsFramework` 中的核心方法**:
    *   `_determine_complexity` 和 `_recruit_experts` 现在都会返回其决策结果以及一个包含该步骤所有细节的日志字典。
    *   `_process_*_query` 方法（处理基础、中等、复杂三种情况）现在会在其返回结果中包含一个名为 `detailed_log` 的字段，里面详细记录了该处理路径下的所有智能体交互和LLM调用日志。

4.  **`MDAgentsFramework.run_query`**:
    *   这是总的协调器，现在它会按顺序收集上述所有方法返回的日志。
    *   最终生成的 JSON 结果文件中，新增了一个顶层字段 `process_log`。这是一个列表，按时间顺序记录了从“判断复杂度”到“招募专家/团队”再到“具体问题处理”的每一步的详尽日志。

### 如何使用这些日志进行分析：

现在，当你运行代码后，每个问题生成的 `qid-result.json` 文件将包含一个非常详细的 `process_log`。你可以加载这个JSON文件，然后：

*   **检查 `determine_complexity` 的日志**：查看Moderator的决策过程是否合理。
*   **分析 `recruit_experts` 的日志**：观察Recruiter是否招募了合适的专家或团队。
*   **深入 `process_intermediate_query` 的 `detailed_log`**：
    *   你可以逐一查看 `initial_opinions` 中每个专家的独立判断。他们的prompt是什么？他们的回答是什么？是否存在有专家一开始就给出了正确答案？
    *   然后查看 `final_synthesis`，决策者收到了哪些汇总信息？它的最终决策是如何做出的？是否忽略了少数派的正确意见？
*   **剖析 `process_advanced_query` 的 `detailed_log`**：
    *   `team_discussions` 让你能进入每个团队内部，观察领导和成员间的微观交互。
    *   你可以追踪一个错误观点是如何在团队内部形成，并最终体现在团队报告中的。
    *   最后，分析 `final_decision_synthesis`，看看最终决策者是如何权衡不同团队（可能相互矛盾）的报告的。
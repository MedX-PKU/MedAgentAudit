#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.." || exit
echo "Current working directory: $(pwd)"
# 设置并发数
MAX_CONCURRENT=4
CURRENT_JOBS=0

# 创建一个临时文件来跟踪正在运行的进程
TEMP_FILE=$(mktemp)
trap "rm -f $TEMP_FILE" EXIT

# 运行命令并管理并发
run_command() {
    local cmd="$1"

    # 检查当前运行的进程数
    CURRENT_JOBS=$(wc -l < $TEMP_FILE)

    # 如果当前运行的进程数达到最大值，等待一个进程完成
    while [ $CURRENT_JOBS -ge $MAX_CONCURRENT ]; do
        for pid in $(cat $TEMP_FILE); do
            if ! kill -0 $pid 2>/dev/null; then
                # 进程已结束，从文件中删除
                grep -v "^$pid$" $TEMP_FILE > ${TEMP_FILE}.new
                mv ${TEMP_FILE}.new $TEMP_FILE
            fi
        done
        CURRENT_JOBS=$(wc -l < $TEMP_FILE)
        if [ $CURRENT_JOBS -ge $MAX_CONCURRENT ]; then
            sleep 1
        fi
    done

    # 运行命令
    echo "Running: $cmd"
    eval "$cmd" &

    # 记录进程ID
    echo $! >> $TEMP_FILE
}

# 定义数据集和任务类型
QA_DATASETS=("MedQA" "PubMedQA" "MedXpertQA-text")
VQA_DATASETS=("PathVQA" "VQA-RAD" "Slake")
QA_LLM=("deepseek-reasoner" "gpt-5.2" "gemini-2.5-flash" "qwen3-8b")
VQA_LLM=("glm-4.6v" "gpt-5.2" "gemini-2.5-flash" "qwen3-vl-8b-thinking")
# this experiment‘s samples' num
num=100
time_stamp="20260112"
qa_type="mc"
AUDITOR="gemini-3-flash-preview"
echo "Starting experiments..."

# 1. ColaCare QA
for dataset in "${QA_DATASETS[@]}"; do
    for qa_llm in "${QA_LLM[@]}"; do
        cmd="python -m medagentaudit.framework.colacare --dataset $dataset --qa_type $qa_type --meta_model $qa_llm --doctor_models $qa_llm $qa_llm $qa_llm --auditor_model $AUDITOR --config_path config.toml --num_samples $num --time_stamp $time_stamp" 
        run_command "$cmd"
    done
done

# 2. ColaCare VQA
for dataset in "${VQA_DATASETS[@]}"; do
    for vqa_llm in "${VQA_LLM[@]}"; do
        cmd="python -m medagentaudit.framework.colacare --dataset $dataset --qa_type $qa_type --meta_model $vqa_llm --doctor_models $vqa_llm $vqa_llm $vqa_llm --auditor_model $AUDITOR --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 3. MedAgent QA
for dataset in "${QA_DATASETS[@]}"; do
    for qa_llm in "${QA_LLM[@]}"; do
        cmd="python -m medagentaudit.framework.medagent --dataset $dataset --qa_type $qa_type --model $qa_llm --meta_model $qa_llm --decision_model $qa_llm --auditor_model $AUDITOR --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 4. MedAgent VQA
for dataset in "${VQA_DATASETS[@]}"; do
    for vqa_llm in "${VQA_LLM[@]}"; do
        cmd="python -m medagentaudit.framework.medagent --dataset $dataset --qa_type $qa_type --model $vqa_llm --meta_model $vqa_llm --decision_model $vqa_llm --auditor_model $AUDITOR --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done


# 5. MDAgents QA
for dataset in "${QA_DATASETS[@]}"; do
    for qa_llm in "${QA_LLM[@]}"; do
        cmd="python -m medagentaudit.framework.mdagents --dataset $dataset --qa_type $qa_type --moderator_model $qa_llm --recruiter_model $qa_llm --agent_model $qa_llm --auditor_model $AUDITOR --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 6. MDAgents VQA
for dataset in "${VQA_DATASETS[@]}"; do
    for vqa_llm in "${VQA_LLM[@]}"; do
        cmd="python -m medagentaudit.framework.mdagents --dataset $dataset --qa_type $qa_type --moderator_model $vqa_llm --recruiter_model $vqa_llm --agent_model $vqa_llm --auditor_model $AUDITOR --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 7. ReConcile QA
for dataset in "${QA_DATASETS[@]}"; do
    for qa_llm in "${QA_LLM[@]}"; do
        cmd="python -m medagentaudit.medqa.multi_agent_reconcile_audit --dataset $dataset --qa_type $qa_type --agents $qa_llm $qa_llm $qa_llm --auditor_model $AUDITOR --config_path config.toml --num_samples $num --time_stamp $time_stamp --max_rounds 3"
        run_command "$cmd"
    done
done

# 8. ReConcile VQA
for dataset in "${VQA_DATASETS[@]}"; do
    for vqa_llm in "${VQA_LLM[@]}"; do
        cmd="python -m medagentaudit.medqa.multi_agent_reconcile_audit --dataset $dataset --qa_type $qa_type --agents $vqa_llm $vqa_llm $vqa_llm --auditor_model $AUDITOR --config_path config.toml --num_samples $num --time_stamp $time_stamp --max_rounds 3"
        run_command "$cmd"
    done
done

# 9. MAC QA
for dataset in "${QA_DATASETS[@]}"; do
    for qa_llm in "${QA_LLM[@]}"; do
        cmd="python -m medagentaudit.medqa.multi_agent_mac_audit --dataset $dataset --qa_type $qa_type --doctor_model $qa_llm --supervisor_model $qa_llm --auditor_model $AUDITOR --config_path config.toml --num_samples $num --time_stamp $time_stamp --max_rounds 3"
        run_command "$cmd"
    done
done

# 10. MAC VQA
for dataset in "${VQA_DATASETS[@]}"; do
    for vqa_llm in "${VQA_LLM[@]}"; do
        cmd="python -m medagentaudit.medqa.multi_agent_mac_audit --dataset $dataset --qa_type $qa_type --doctor_model $vqa_llm --supervisor_model $vqa_llm --auditor_model $AUDITOR --config_path config.toml --num_samples $num --time_stamp $time_stamp --max_rounds 3"
        run_command "$cmd"
    done
done

# 11. HealthcareAgent QA
for dataset in "${QA_DATASETS[@]}"; do
    for qa_llm in "${QA_LLM[@]}"; do
        cmd="python -m medagentaudit.medqa.multi_agent_healthcareagent_audit --dataset $dataset --qa_type $qa_type --model $qa_llm --auditor_model $AUDITOR --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 12. HealthcareAgent VQA
for dataset in "${VQA_DATASETS[@]}"; do
    for vqa_llm in "${VQA_LLM[@]}"; do
        cmd="python -m medagentaudit.medqa.multi_agent_healthcareagent_audit --dataset $dataset --qa_type $qa_type --model $vqa_llm --auditor_model $AUDITOR --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done
# 等待所有任务完成
wait

echo "All experiments completed!"
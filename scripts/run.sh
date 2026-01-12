#!/bin/bash

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

# 为每个数据集定义可用的qa_type
declare -A DATASET_QA_TYPES
DATASET_QA_TYPES[MedQA]="mc"
DATASET_QA_TYPES[PubMedQA]="mc"
DATASET_QA_TYPES[PathVQA]="mc"
DATASET_QA_TYPES[VQA-RAD]="mc"
DATASET_QA_TYPES[Slake]="mc"
DATASET_QA_TYPES[MedXpertQA-text]="mc"
# this experiment‘s samples' num
num=100
time_stamp="20260111"
echo "Starting experiments..."

# 1. ColaCare QA deepseek-reasoner
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_colacare_audit --dataset $dataset --qa_type $qa_type --meta_model deepseek-reasoner --doctor_models deepseek-reasoner deepseek-reasoner deepseek-reasoner --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp" 
        run_command "$cmd"
    done
done

# 2. ColaCare QA gpt-5.2
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_colacare_audit --dataset $dataset --qa_type $qa_type --meta_model gpt-5.2 --doctor_models gpt-5.2 gpt-5.2 gpt-5.2 --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 3. ColaCare QA gemini-2.5-flash
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_colacare_audit --dataset $dataset --qa_type $qa_type --meta_model gemini-2.5-flash --doctor_models gemini-2.5-flash gemini-2.5-flash gemini-2.5-flash --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 4. ColaCare QA qwen3-8b
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_colacare_audit --dataset $dataset --qa_type $qa_type --meta_model qwen3-8b --doctor_models qwen3-8b qwen3-8b qwen3-8b --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 5. ColaCare VQA glm-4.6v
for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_colacare_audit --dataset $dataset --qa_type $qa_type --meta_model glm-4.6v --doctor_models glm-4.6v glm-4.6v glm-4.6v --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 6. ColaCare VQA gpt-5.2
for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_colacare_audit --dataset $dataset --qa_type $qa_type --meta_model gpt-5.2 --doctor_models gpt-5.2 gpt-5.2 gpt-5.2 --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 7. ColaCare VQA gemini-2.5-flash
for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_colacare_audit --dataset $dataset --qa_type $qa_type --meta_model gemini-2.5-flash --doctor_models gemini-2.5-flash gemini-2.5-flash gemini-2.5-flash --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 8. ColaCare VQA qwen3-vl-8b-thinking
for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_colacare_audit --dataset $dataset --qa_type $qa_type --meta_model qwen3-vl-8b-thinking --doctor_models qwen3-vl-8b-thinking qwen3-vl-8b-thinking qwen3-vl-8b-thinking --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 9. MedAgent QA deepseek-reasoner
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_medagent_audit --dataset $dataset --qa_type $qa_type --model deepseek-reasoner --meta_model deepseek-reasoner --decision_model deepseek-reasoner --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 10. MedAgent QA gpt-5.2
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_medagent_audit --dataset $dataset --qa_type $qa_type --model gpt-5.2 --meta_model gpt-5.2 --decision_model gpt-5.2 --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 11. MedAgent QA gemini-2.5-flash
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_medagent_audit --dataset $dataset --qa_type $qa_type --model gemini-2.5-flash --meta_model gemini-2.5-flash --decision_model gemini-2.5-flash --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 12. MedAgent QA qwen3-8b
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_medagent_audit --dataset $dataset --qa_type $qa_type --model qwen3-8b --meta_model qwen3-8b --decision_model qwen3-8b --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 13. MedAgent VQA glm-4.6v
for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_medagent_audit --dataset $dataset --qa_type $qa_type --model glm-4.6v --meta_model glm-4.6v --decision_model glm-4.6v --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 14. MedAgent VQA gpt-5.2
for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_medagent_audit --dataset $dataset --qa_type $qa_type --model gpt-5.2 --meta_model gpt-5.2 --decision_model gpt-5.2 --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 15. MedAgent VQA gemini-2.5-flash
for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_medagent_audit --dataset $dataset --qa_type $qa_type --model gemini-2.5-flash --meta_model gemini-2.5-flash --decision_model gemini-2.5-flash --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 16. MedAgent VQA qwen3-vl-8b-thinking
for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_medagent_audit --dataset $dataset --qa_type $qa_type --model qwen3-vl-8b-thinking --meta_model qwen3-vl-8b-thinking --decision_model qwen3-vl-8b-thinking --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 17. MDAgents QA deepseek-reasoner
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_mdagents_audit --dataset $dataset --qa_type $qa_type --moderator_model deepseek-reasoner --recruiter_model deepseek-reasoner --agent_model deepseek-reasoner --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_mdagents_audit --dataset $dataset --qa_type $qa_type --moderator_model glm-4.6v --recruiter_model gpt-5.2 --agent_model gemini-2.5-flash --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

# 4. ReConcile
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_reconcile_audit --dataset $dataset --qa_type $qa_type --agents deepseek-reasoner gpt-5.2 gemini-2.5-flash --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp --max_rounds 3"
        run_command "$cmd"
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_reconcile_audit --dataset $dataset --qa_type $qa_type --agents glm-4.6v gpt-5.2 gemini-2.5-flash --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp --max_rounds 3"
        run_command "$cmd"
    done
done

# 5. MAC
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_mac_audit --dataset $dataset --qa_type $qa_type --doctor_model deepseek-reasoner --supervisor_model gpt-5.2 --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp --max_rounds 3"
        run_command "$cmd"
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_mac_audit --dataset $dataset --qa_type $qa_type --doctor_model glm-4.6v --supervisor_model gpt-5.2 --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp --max_rounds 3"
        run_command "$cmd"
    done
done

# 6. HealthcareAgent
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_healthcareagent_audit --dataset $dataset --qa_type $qa_type --model deepseek-reasoner --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_healthcareagent_audit --dataset $dataset --qa_type $qa_type --model glm-4.6v --auditor_model gemini-3-flash-preview --config_path config.toml --num_samples $num --time_stamp $time_stamp"
        run_command "$cmd"
    done
done
# 等待所有任务完成
wait

echo "All experiments completed!"
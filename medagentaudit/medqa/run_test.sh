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
QA_DATASETS=("MedQA") # 测试每个框架qa用medqa跑一个case，vqa用PathVQA 跑一个case，一共2*6 = 12个，然后domain agent要换三个模型，确保都能跑
VQA_DATASETS=("PathVQA")

# 为每个数据集定义可用的qa_type
declare -A DATASET_QA_TYPES
DATASET_QA_TYPES[MedQA]="mc"
DATASET_QA_TYPES[PubMedQA]="mc"
DATASET_QA_TYPES[PathVQA]="mc"
DATASET_QA_TYPES[VQA-RAD]="mc"

# this experiment‘s samples' num
num=1

echo "Starting experiments..."

# 1. ColaCare
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_colacare_full_log_add_mechanism --dataset $dataset --qa_type $qa_type --meta_model deepseek-reasoner --doctor_models deepseek-reasoner gpt-5.1 gemini-2.5-flash --auditor_model gemini-3-pro-preview --config_path config.toml --num_samples $num --test_mode True" 
        run_command "$cmd"
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_colacare_full_log_add_mechanism --dataset $dataset --qa_type $qa_type --meta_model glm-4.6v --doctor_models glm-4.6v gpt-5.1 gemini-2.5-flash --auditor_model gemini-3-pro-preview --config_path config.toml --num_samples $num --test_mode True"
        run_command "$cmd"
    done
done

# 2. MedAgent
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_medagent_full_log_add_mechanism --dataset $dataset --qa_type $qa_type --model deepseek-reasoner --meta_model gpt-5.1 --decision_model gemini-2.5-flash --auditor_model gemini-3-pro-preview --config_path config.toml --num_samples $num --test_mode True"
        run_command "$cmd"
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_medagent_full_log_add_mechanism --dataset $dataset --qa_type $qa_type --model glm-4.6v --meta_model gpt-5.1 --decision_model gemini-2.5-flash --auditor_model gemini-3-pro-preview --config_path config.toml --num_samples $num --test_mode True"
        run_command "$cmd"
    done
done

# 3. MDAgents
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_mdagents_full_log_add_mechanism --dataset $dataset --qa_type $qa_type --moderator_model deepseek-reasoner --recruiter_model gpt-5.1 --agent_model gemini-2.5-flash --auditor_model gemini-3-pro-preview --config_path config.toml --num_samples $num --test_mode True"
        run_command "$cmd"
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_mdagents_full_log_add_mechanism --dataset $dataset --qa_type $qa_type --moderator_model glm-4.6v --recruiter_model gpt-5.1 --agent_model gemini-2.5-flash --auditor_model gemini-3-pro-preview --config_path config.toml --num_samples $num --test_mode True"
        run_command "$cmd"
    done
done

# 4. ReConcile
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_reconcile_full_log_add_mechanism --dataset $dataset --qa_type $qa_type --agents deepseek-reasoner gpt-5.1 gemini-2.5-flash --auditor_model gemini-3-pro-preview --config_path config.toml --num_samples $num --test_mode True --max_rounds 3"
        run_command "$cmd"
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_reconcile_full_log_add_mechanism --dataset $dataset --qa_type $qa_type --agents glm-4.6v gpt-5.1 gemini-2.5-flash --auditor_model gemini-3-pro-preview --config_path config.toml --num_samples $num --test_mode True --max_rounds 3"
        run_command "$cmd"
    done
done

# 5. MAC
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_mac_add_mechanism --dataset $dataset --qa_type $qa_type --doctor_model deepseek-reasoner --supervisor_model gpt-5.1 --auditor_model gemini-3-pro-preview --config_path config.toml --num_samples $num --test_mode True --max_rounds 3"
        run_command "$cmd"
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_mac_add_mechanism --dataset $dataset --qa_type $qa_type --doctor_model glm-4.6v --supervisor_model gpt-5.1 --auditor_model gemini-3-pro-preview --config_path config.toml --num_samples $num --test_mode True --max_rounds 3"
        run_command "$cmd"
    done
done

# 6. HealthcareAgent
for dataset in "${QA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_healthcareagent_add_mechanism --dataset $dataset --qa_type $qa_type --doctor_model deepseek-reasoner --auditor_model gemini-3-pro-preview --config_path config.toml --num_samples $num --test_mode True --max_rounds 3"
        run_command "$cmd"
    done
done

for dataset in "${VQA_DATASETS[@]}"; do
    for qa_type in ${DATASET_QA_TYPES[$dataset]}; do
        cmd="python -m medagentaudit.medqa.multi_agent_healthcareagent_add_mechanism --dataset $dataset --qa_type $qa_type --doctor_model glm-4.6v --auditor_model gemini-3-pro-preview --config_path config.toml --num_samples $num --test_mode True --max_rounds 3"
        run_command "$cmd"
    done
done
# 等待所有任务完成
wait

echo "All experiments completed!"
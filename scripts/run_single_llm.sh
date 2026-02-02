:<< 'END_OF_DOCS'
./scripts/run_single_llm.sh
this file is used to run single LLM experiments in parallel
END_OF_DOCS

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
QA_LLM=("deepseek-reasoner" "gpt-5.2" "gemini-3-flash-preview" "qwen3-8b")
VQA_LLM=("glm-4.6v" "gpt-5.2" "gemini-3-flash-preview" "qwen3-vl-8b-thinking")
# this experiment‘s samples' num
num=100
time_stamp="20260202"
config_path="config.toml"
echo "Starting experiments..."

# 1. QA
for dataset in "${QA_DATASETS[@]}"; do
    for qa_llm in "${QA_LLM[@]}"; do
        cmd="python -m medagentaudit.framework.single_llm --dataset $dataset --config_path $config_path --num_samples $num --time_stamp $time_stamp --model_key $qa_llm" 
        run_command "$cmd"
    done
done

# 2. VQA
for dataset in "${VQA_DATASETS[@]}"; do
    for vqa_llm in "${VQA_LLM[@]}"; do
        cmd="python -m medagentaudit.framework.single_llm --dataset $dataset --config_path $config_path --num_samples $num --time_stamp $time_stamp --model_key $vqa_llm"
        run_command "$cmd"
    done
done

wait

echo "All experiments completed!"
:<< 'END_OF_DOCS'
./scripts/run_open_coding.sh
this file is used to run open-coding experiments in parallel
END_OF_DOCS

#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.." || exit
echo "Current working directory: $(pwd)"
# 设置并发数
MAX_CONCURRENT=20
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

VQA_DATASETS=("PathVQA" "VQA-RAD" "SLAKE")

QA_LLM=("deepseek-reasoner")

VQA_LLM=("gemini-3-flash-preview")
echo "Starting experiments..."

# 1. ColaCare QA fixed
for dataset in "${QA_DATASETS[@]}"; do
    for qa_llm in "${QA_LLM[@]}"; do
        cmd="python -m scripts.run_open_coding --dataset $dataset --llm $qa_llm --mas colacare" 
        run_command "$cmd"
    done
done

# 2. ColaCare VQA fixed
for dataset in "${VQA_DATASETS[@]}"; do
    for vqa_llm in "${VQA_LLM[@]}"; do
        cmd="python -m scripts.run_open_coding --dataset $dataset --llm $vqa_llm --mas colacare" 
        run_command "$cmd"
    done
done

# 3. MedAgent QA fixed
for dataset in "${QA_DATASETS[@]}"; do
    for qa_llm in "${QA_LLM[@]}"; do
        cmd="python -m scripts.run_open_coding --dataset $dataset --llm $qa_llm --mas medagent" 
        run_command "$cmd"
    done
done

# 4. MedAgent VQA fixed
for dataset in "${VQA_DATASETS[@]}"; do
    for vqa_llm in "${VQA_LLM[@]}"; do
        cmd="python -m scripts.run_open_coding --dataset $dataset --llm $vqa_llm --mas medagent" 
        run_command "$cmd"
    done
done


# 5. MDAgents QA fixed
for dataset in "${QA_DATASETS[@]}"; do
    for qa_llm in "${QA_LLM[@]}"; do
        cmd="python -m scripts.run_open_coding --dataset $dataset --llm $qa_llm --mas mdagents"
        run_command "$cmd"
    done
done

# 6. MDAgents VQA fixed
for dataset in "${VQA_DATASETS[@]}"; do
    for vqa_llm in "${VQA_LLM[@]}"; do
        cmd="python -m scripts.run_open_coding --dataset $dataset --llm $vqa_llm --mas mdagents"
        run_command "$cmd"
    done
done

# 7. ReConcile QA fixed
for dataset in "${QA_DATASETS[@]}"; do
    for qa_llm in "${QA_LLM[@]}"; do
        cmd="python -m scripts.run_open_coding --dataset $dataset --llm $qa_llm --mas reconcile"
        run_command "$cmd"
    done
done

# 8. ReConcile VQA fixed
for dataset in "${VQA_DATASETS[@]}"; do
    for vqa_llm in "${VQA_LLM[@]}"; do
        cmd="python -m scripts.run_open_coding --dataset $dataset --llm $vqa_llm --mas reconcile"
        run_command "$cmd"
    done
done

# 9. MAC QA fixed
for dataset in "${QA_DATASETS[@]}"; do
    for qa_llm in "${QA_LLM[@]}"; do
        cmd="python -m scripts.run_open_coding --dataset $dataset --llm $qa_llm --mas mac"
        run_command "$cmd"
    done
done

# 10. MAC VQA fixed
for dataset in "${VQA_DATASETS[@]}"; do
    for vqa_llm in "${VQA_LLM[@]}"; do
        cmd="python -m scripts.run_open_coding --dataset $dataset --llm $vqa_llm --mas mac"
        run_command "$cmd"
    done
done

# 11. HealthcareAgent QA
for dataset in "${QA_DATASETS[@]}"; do
    for qa_llm in "${QA_LLM[@]}"; do
        cmd="python -m scripts.run_open_coding --dataset $dataset --llm $qa_llm --mas healthcareagent"
        run_command "$cmd"
    done
done

# 12. HealthcareAgent VQA
for dataset in "${VQA_DATASETS[@]}"; do
    for vqa_llm in "${VQA_LLM[@]}"; do
        cmd="python -m scripts.run_open_coding --dataset $dataset --llm $vqa_llm --mas healthcareagent"
        run_command "$cmd"
    done
done
# 等待所有任务完成
wait

echo "All experiments completed!"

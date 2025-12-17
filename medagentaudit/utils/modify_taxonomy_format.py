import json
import re
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve()))
from typing import Dict, Any, Union
from get_logger import get_logger

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
logs_dir = project_root / "logs" / "annotation"
logger_file_name = project_root / "logs" / "annotation" / "20251217_rm_hyphen_and_text_from_taxonomy_code.log"
logger_file_name.parent.mkdir(parents=True, exist_ok=True) # ensure log directory exists
logger = get_logger(name='20251217_rm_hyphen_and_text_from_taxonomy_code', file_name=logger_file_name)

def clean_code_string(code_str: str) -> str:
    """
    使用正则表达式提取纯数字编码。
    Ex: "1.1.1-Visual Hallucination" -> "1.1.1"
    Ex: "2.1" -> "2.1"
    """
    if not code_str:
        return code_str
    
    # 匹配开头的一串数字和点号
    match = re.match(r"^([\d\.]+)", code_str)
    if match:
        return match.group(1)
    
    # 如果没匹配到（虽然理论上不应该发生），返回原字符串
    return code_str

def process_single_file(file_path: Path) -> bool:
    """
    处理单个 JSON 文件。如果发生了修改，返回 True，否则返回 False。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False

        # 1. 修正 primary_classification
        if "primary_classification" in data and isinstance(data["primary_classification"], dict):
            raw_code = data["primary_classification"].get("code", "")
            clean_code = clean_code_string(raw_code)
            
            if raw_code != clean_code:
                data["primary_classification"]["code"] = clean_code
                modified = True

        # 2. 修正 secondary_classifications (List)
        if "secondary_classifications" in data and isinstance(data["secondary_classifications"], list):
            for item in data["secondary_classifications"]:
                if isinstance(item, dict):
                    raw_code = item.get("code", "")
                    clean_code = clean_code_string(raw_code)
                    
                    if raw_code != clean_code:
                        item["code"] = clean_code
                        modified = True

        # 3. 如果有变动，写回文件
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
            
    except json.JSONDecodeError:
        logger.error(f"JSON 格式错误，跳过文件: {file_path}")
    except Exception as e:
        logger.error(f"处理文件出错 {file_path}: {e}")
        
    return False

def main():
    logger.info("开始执行 Taxonomy Code 格式清洗任务...")
    # 使用 rglob 递归查找所有 .json 文件
    json_files = list(logs_dir.rglob("*.json"))
    total_files = len(json_files)
    modified_count = 0
    
    logger.info(f"共发现 {total_files} 个 JSON 文件，开始处理...")

    for json_file in json_files:
        if process_single_file(json_file):
            modified_count += 1
            # 仅在修改时打印，避免日志刷屏
            # logger.info(f"Fixed: {json_file.name}")

    logger.info("="*30)
    logger.info("处理完成 Summary:")
    logger.info(f"扫描文件总数: {total_files}")
    logger.info(f"修正文件数量: {modified_count}")
    logger.info("="*30)

if __name__ == "__main__":
    main()
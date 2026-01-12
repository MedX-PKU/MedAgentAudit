import os
import fnmatch
import argparse

# --- 配置区 ---
# 在这里添加你想要忽略的文件或目录的匹配模式，类似于 .gitignore
# 使用 Unix shell 风格的通配符 (*, ?, [a-z], etc.)

# 忽略文件的模式
IGNORE_FILE_PATTERNS = [
    '*.pyc',
    '*.tmp',
    '*.log',
    '*.DS_Store',
    'package-lock.json',
    '.env',
    '.gitignore',
    '*.ipynb',
    'uv.lock',
    '.pre-commit-config.yaml',
    'CONTRIBUTION.md',
    # 'CLAUDE.md',
    'GEMINI.md',
    'TASK.md',
    'prompt.md',
    '*.example',
    ### Start Alita
    '*.json',
    '*.jsonl',
    '*.cpp',
    'pnpm-lock.yaml',
    'benchmark_results.json',
    'mcp_tools_registry.json',
    ### End Alita
]

# 忽略目录的模式
IGNORE_DIR_PATTERNS = [
    '.git',
    '__pycache__',
    'node_modules',
    'venv',
    '.venv',
    '.vscode',
    'build',
    'dist',
    'docs',
    'tmp',
    'prompts',
    'scripts',
    'logs',
    'mnist',
    ### Start Biomni
    'biomni_env',
    'tutorials',
    'biorxiv_scripts',
    ### End Biomni
    ### Start Alita
    ### End Alita
    ### Start AlphaEvolve
    'tests',
    'workspace',
    'benchmark_results',
    'datasets',
    'healthflow_datasets',
    '.claude',
    ### End AlphaEvolve
    'assets'
]

# 文件扩展名到 Markdown 语言标识符的映射
# 可根据需要自行添加
LANG_MAP = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.java': 'java',
    '.c': 'c',
    '.cpp': 'cpp',
    '.cs': 'csharp',
    '.go': 'go',
    '.rs': 'rust',
    '.rb': 'ruby',
    '.php': 'php',
    '.kt': 'kotlin',
    '.swift': 'swift',
    '.html': 'html',
    '.css': 'css',
    '.scss': 'scss',
    '.json': 'json',
    '.xml': 'xml',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.md': 'markdown',
    '.sh': 'shell',
    '.bat': 'batch',
    'Dockerfile': 'dockerfile',
}
# --- 配置区结束 ---


def get_lang_identifier(filename):
    """根据文件名获取Markdown代码块的语言标识符"""
    # 首先检查完整文件名是否匹配（例如 Dockerfile）
    if filename in LANG_MAP:
        return LANG_MAP[filename]
    # 然后根据扩展名匹配
    _, ext = os.path.splitext(filename)
    return LANG_MAP.get(ext, '') # 如果找不到，返回空字符串

def should_ignore(name, patterns):
    """检查名称是否匹配任何忽略模式"""
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False

def consolidate_code(source_dir, output_file):
    """
    将指定目录下的所有代码递归地写入一个文件，并用Markdown代码块包裹。

    :param source_dir: 要处理的源文件夹路径。
    :param output_file: 输出文件的路径。
    """
    if not os.path.isdir(source_dir):
        print(f"错误：源目录 '{source_dir}' 不存在或不是一个目录。")
        return

    print(f"开始处理目录: {os.path.abspath(source_dir)}")
    print(f"将输出到文件: {os.path.abspath(output_file)}")

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for root, dirs, files in os.walk(source_dir, topdown=True):
                dirs[:] = [d for d in dirs if not should_ignore(d, IGNORE_DIR_PATTERNS)]

                # 对文件进行排序，以保证输出顺序的一致性
                files.sort()

                for filename in files:
                    if should_ignore(filename, IGNORE_FILE_PATTERNS):
                        continue

                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, source_dir)

                    # 写入文件头，现在作为Markdown的标题
                    outfile.write(f"### `{relative_path}`\n\n")
                    print(f"正在添加: {relative_path}")

                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            lang = get_lang_identifier(filename)

                            # 写入被Markdown代码块包裹的内容
                            outfile.write(f"```{lang}\n")
                            outfile.write(content)
                            outfile.write("\n```\n\n") # 确保代码块在新的一行结束

                    except UnicodeDecodeError:
                        outfile.write(f"`--- SKIPPED (无法读取的文件，可能是二进制文件): {relative_path} ---`\n\n")
                        print(f"  -> 已跳过 (非文本文件): {relative_path}")
                    except Exception as e:
                        outfile.write(f"`--- SKIPPED (读取文件时出错: {e}): {relative_path} ---`\n\n")
                        print(f"  -> 读取出错已跳过: {relative_path} (错误: {e})")

    except IOError as e:
        print(f"错误：无法写入到输出文件 '{output_file}'. {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

    print("\n处理完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将指定目录下的所有代码（递归）合并到一个Markdown文件中，每个文件内容都用代码块包裹。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("source_dir", help="要处理的源文件夹的路径。")
    parser.add_argument("output_file", help="合并后输出的文件的路径（建议使用.md扩展名）。")

    args = parser.parse_args()
    consolidate_code(args.source_dir, args.output_file)
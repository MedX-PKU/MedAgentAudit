import sys
import os
import time
class DualLogger:
    def __init__(self, filepath, stream, max_logs = 2):
        self.terminal = stream
        self.filepath = os.path.abspath(filepath)
        log_dir = os.path.dirname(self.filepath)
        self.log = open(self.filepath, "a", encoding="utf-8")

        # every start we write a separator line with timestamp
        start_msg = f"\n\n{'='*30} NEW SESSION: {time.strftime('%Y-%m-%d %H:%M:%S')} {'='*30}\n"
        self.write_to_file_only(start_msg)

    def write(self, message):
        # 写入终端（保持原有显示）
        self.terminal.write(message)
        # 写入文件
        self.log.write(message)
        # 强制刷新缓冲区，确保程序崩溃时也能看到最后的日志
        self.log.flush()  

    def write_to_file_only(self, message):
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # 用于兼容 flush 调用
        self.terminal.flush()
        self.log.flush()
        
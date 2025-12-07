import sys
import os
import time
class DualLogger:
    def __init__(self, filepath, stream):
        self.terminal = stream
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        # 写入终端（保持原有显示）
        self.terminal.write(message)
        # 写入文件
        self.log.write(message)
        # 强制刷新缓冲区，确保程序崩溃时也能看到最后的日志
        self.log.flush()  

    def flush(self):
        # 用于兼容 flush 调用
        self.terminal.flush()
        self.log.flush()
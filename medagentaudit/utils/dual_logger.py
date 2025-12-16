import sys
import os
import time
class DualLogger:
    def __init__(self, filepath, stream, max_logs = 2):
        self.terminal = stream
        self.filepath = os.path.abspath(filepath)
        log_dir = os.path.dirname(self.filepath)
        self.log = open(self.filepath, "a", encoding="utf-8")
        self._rotate_logs(log_dir, max_logs)
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
    def _rotate_logs(self, log_dir, max_logs):
        try:
            # 获取该目录下所有 .log 文件
            all_logs = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.log')]
            
            # 如果文件数超过限制
            if len(all_logs) > max_logs:
                # 按修改时间排序：最旧的在前面
                all_logs.sort(key=os.path.getmtime)
                
                # 需要删除的数量
                num_to_delete = len(all_logs) - max_logs
                
                # 删除最旧的 num_to_delete 个文件
                for i in range(num_to_delete):
                    log_to_remove = all_logs[i]
                    
                    # 安全检查：千万别删掉当前正在写的这个文件
                    # (虽然按时间排序当前文件通常在最后，但加个判断更保险)
                    if os.path.abspath(log_to_remove) != self.filepath:
                        try:
                            os.remove(log_to_remove)
                            print(f"!!! [Log Rotation] Limit ({max_logs}) reached. Deleted old log: {log_to_remove} !!!")
                        except OSError as e:
                            print(f"!!! [Log Rotation] Error deleting {log_to_remove}: {e} !!!")
        except Exception as e:
            # 防止清理逻辑报错导致程序崩溃
            print(f"!!! [Log Rotation] Failed to rotate logs: {e} !!!")
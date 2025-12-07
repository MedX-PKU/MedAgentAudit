from typing import List, Dict, Any

class KEU:
    def __init__(self, keu_id: str, content: str, source_agent: str, round_introduced: int):
        self.keu_id: str = keu_id
        self.content: str = content
        self.source_agent: str = source_agent
        self.round_introduced: int = round_introduced
        
        # --- 新增和修改的属性 ---
        self.is_key: bool = False  # 是否为关键证据，将由Auditor在第一轮确定
        
        # 记录引用历史（支持或中性引用）
        self.cited_by: List[Dict[str, Any]] = []  # e.g., {'agent_id': 'meta', 'round': 1, 'action': 'synthesis'}
        
        # 记录反驳历史
        self.rebuttals: List[Dict[str, Any]] = [] # e.g., {'agent_id': 'doctor_2', 'round': 1, 'reason': '...'}
        
        # 记录在各环节的出现情况
        self.present_in_synthesis: Dict[int, bool] = {} # {round_num: True/False}
        self.present_in_final_decision: bool = False

    def to_dict(self):
        # self.__dict__ 可以直接将所有实例属性转为字典
        return self.__dict__
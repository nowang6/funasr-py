import time
from typing import Dict, Optional
from fastapi import WebSocket
from src.logger import logger


# ==================== WebSocket 会话管理 ====================
class SessionManager:
    """管理 WebSocket 会话"""
    
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        self.stats = {
            'active_sessions': 0,
            'total_requests': 0,
            'queue_size': 0
        }
        
    def create_session(self, session_id: str, websocket: WebSocket):
        """创建新会话"""
        self.sessions[session_id] = {
            'websocket': websocket,
            'created_at': time.time(),
            'first_pass_count': 0,
            'second_pass_count': 0,
            'audio_buffer': b'',
            'is_speaking': False,
            # 音频帧缓冲
            'frames': [],
            'frames_asr': [],
            'frames_asr_online': [],
            # VAD 相关状态
            'vad_pre_idx': 0,
            'speech_start': False,
            'speech_end_i': -1,
            # 模型状态字典
            'status_dict_asr': {},
            'status_dict_asr_online': {"cache": {}, "is_final": False},
            'status_dict_vad': {"cache": {}, "is_final": False},
            'status_dict_punc': {"cache": {}},
            # 配置参数
            'chunk_interval': 10,
            'chunk_size': [5, 10, 5],
            'mode': '2pass',
            'wav_name': 'microphone'
        }
        self.stats['active_sessions'] = len(self.sessions)
        logger.info(f"创建会话: {session_id}, 当前活跃会话数: {self.stats['active_sessions']}")
        
    def remove_session(self, session_id: str):
        """移除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.stats['active_sessions'] = len(self.sessions)
            logger.info(f"移除会话: {session_id}, 当前活跃会话数: {self.stats['active_sessions']}")
            
    def get_session(self, session_id: str) -> Optional[dict]:
        """获取会话"""
        return self.sessions.get(session_id)

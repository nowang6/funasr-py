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
        current_time = time.time()
        current_time_ms = int(current_time * 1000)
        self.sessions[session_id] = {
            'websocket': websocket,
            'created_at': current_time,
            'last_activity': current_time,  # 最后活动时间
            'send_failed': False,  # 标记发送是否失败
            'is_closing': False,  # 标记会话是否正在关闭
            'first_pass_count': 0,
            'second_pass_count': 0,
            'audio_buffer': b'',
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
            'wav_name': 'microphone',
            # 时间戳相关（用于响应消息）
            'start_time_ms': current_time_ms,  # 会话开始时间（毫秒）
            'last_bg': 0  # 上次结束时间，用于计算下次的bg
        }
        self.stats['active_sessions'] = len(self.sessions)
        logger.info(f"创建会话: {session_id}, 当前活跃会话数: {self.stats['active_sessions']}")
        
    def remove_session(self, session_id: str):
        """移除会话"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # 清理会话中的大对象，释放内存
            session['frames'] = []
            session['frames_asr'] = []
            session['frames_asr_online'] = []
            session['audio_buffer'] = b''
            
            del self.sessions[session_id]
            self.stats['active_sessions'] = len(self.sessions)
            logger.info(f"移除会话: {session_id}, 当前活跃会话数: {self.stats['active_sessions']}")
            
    def get_session(self, session_id: str) -> Optional[dict]:
        """获取会话"""
        return self.sessions.get(session_id)
    
    def cleanup_timeout_sessions(self, timeout_seconds: int = 300) -> int:
        """
        清理超时的会话
        
        Args:
            timeout_seconds: 超时时间（秒），默认5分钟
            
        Returns:
            清理的会话数量
        """
        current_time = time.time()
        timeout_sessions = []
        
        for session_id, session in self.sessions.items():
            # 检查最后活动时间是否超时
            if current_time - session.get('last_activity', session['created_at']) > timeout_seconds:
                timeout_sessions.append(session_id)
        
        # 移除超时会话
        for session_id in timeout_sessions:
            logger.warning(f"会话超时，自动清理: {session_id}")
            self.remove_session(session_id)
        
        return len(timeout_sessions)

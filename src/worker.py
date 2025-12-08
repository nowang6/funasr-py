import asyncio
import json
import logging
import time
from datetime import datetime
from multiprocessing import Process, Queue, Manager
from typing import Dict, Optional
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
from funasr import AutoModel

from src.logger import logger


# ==================== 模型加载和推理进程 ====================
class ASRWorker:
    """ASR 推理工作进程"""
    
    def __init__(self, worker_id: int, task_queue: Queue, result_queue: Queue):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.model_asr = None
        self.model_asr_streaming = None
        self.model_vad = None
        self.model_punc = None
        
    def load_model(self):
        """加载 ASR 模型（这里需要替换为实际的模型加载代码）"""
        logger.info(f"Worker {self.worker_id}: 开始加载模型...")
        ASR_MODEL="models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        ASR_MODEL_ONLINE="models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
        VAD_MODEL="models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        PUNC_MODEL="models/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
        ITN_MODEL="models/fst_itn_zh"
        LM_MODEL="models/speech_ngram_lm_zh-cn-ai-wesp-fst"

        self.model_asr = AutoModel(
            model=ASR_MODEL,
            disable_pbar=True,
            disable_log=True,
            disable_update=True,
        )
        # asr
        self.model_asr_streaming = AutoModel(
            model=ASR_MODEL_ONLINE,
            disable_pbar=True,
            disable_log=True,
            disable_update=True,
        )
        # vad
        self.model_vad = AutoModel(
            model=VAD_MODEL,
            disable_pbar=True,
            disable_log=True,
            disable_update=True,
            # chunk_size=60,
        )

        # punc
        self.model_punc = AutoModel(
            model=PUNC_MODEL,
            disable_pbar=True,
            disable_log=True,
            disable_update=True,
        )
        
        # language model (FST format)
        # 参考 C++ InitLm 函数，FST 语言模型不使用 AutoModel 加载
        # 而是在推理时通过 lm_file 参数传入
        try:
            import os
            lm_file = os.path.join(LM_MODEL, "TLG.fst")
            lm_cfg_file = os.path.join(LM_MODEL, "config.yaml")
            lex_file = os.path.join(LM_MODEL, "lexicon.txt")
            
            # 检查文件是否存在
            if os.path.exists(lm_file):
                self.model_lm = lm_file  # 保存 FST 文件路径
                self.lm_cfg_file = lm_cfg_file if os.path.exists(lm_cfg_file) else None
                self.lex_file = lex_file if os.path.exists(lex_file) else None
                logger.info(f"Worker {self.worker_id}: Successfully load lm file {lm_file}")
            else:
                self.model_lm = None
                logger.warning(f"Worker {self.worker_id}: LM file not found: {lm_file}")
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error when load lm file: {e}")
            self.model_lm = None

        
        logger.info(f"Worker {self.worker_id}: 模型加载完成")
        
    def transcribe_online(self, audio_data: bytes, session_id: str, status_dict: dict) -> dict:
        """执行在线流式语音识别（第一遍快速识别）
        
        返回结果中包含更新后的status_dict_asr_online字段
        """
        try:
            if len(audio_data) > 0:
                rec_result = self.model_asr_streaming.generate(
                    input=audio_data, 
                    **status_dict
                )[0]
                
                return {
                    "session_id": session_id,
                    "text": rec_result.get("text", ""),
                    "mode": "online",
                    "is_final": False,
                    "timestamp": time.time(),
                    "status_dict_asr_online": status_dict  # 返回更新后的status_dict
                }
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: 在线识别错误: {e}")
            
        return {
            "session_id": session_id,
            "text": "",
            "mode": "online",
            "is_final": False,
            "timestamp": time.time(),
            "status_dict_asr_online": status_dict
        }
    
    def transcribe_offline(self, audio_data: bytes, session_id: str, status_dict_asr: dict, status_dict_punc: dict, is_final: bool = True) -> dict:
        """执行离线语音识别（第二遍精细识别 + 标点）
        
        参数:
            audio_data: 音频数据
            session_id: 会话ID
            status_dict_asr: ASR状态字典
            status_dict_punc: 标点状态字典
            is_final: 是否是最后一帧（对应C++的input_finished）
        """
        try:
            if len(audio_data) > 0:
                # 执行离线 ASR
                # 参考 C++ 代码: if (lm_ == nullptr) { GreedySearch } else { BeamSearch }
                generate_kwargs = status_dict_asr.copy()
                
                # 根据是否有语言模型选择解码策略
                if self.model_lm is not None:
                    # 有语言模型 -> 使用 BeamSearch (对应C++ BeamSearch逻辑)
                    generate_kwargs['beam_search'] = True
                    generate_kwargs['beam_size'] = 10  # 默认束宽度
                    generate_kwargs['lm_file'] = self.model_lm
                    if hasattr(self, 'lm_cfg_file') and self.lm_cfg_file:
                        generate_kwargs['lm_cfg_file'] = self.lm_cfg_file
                    if hasattr(self, 'lex_file') and self.lex_file:
                        generate_kwargs['lex_file'] = self.lex_file
                    # 语言模型权重
                    generate_kwargs['lm_weight'] = 0.15
                    generate_kwargs['decoding_ctc_weight'] = 0.5
                else:
                    # 没有语言模型 -> 使用 GreedySearch (对应C++ GreedySearch逻辑)
                    generate_kwargs['beam_search'] = False
                
                # 对应C++的 input_finished 标志
                generate_kwargs['is_final'] = is_final
                
                # 对应C++代码中的timestamp处理(us_alphas, us_peaks)
                # Python FunASR会在结果中返回timestamp信息
                rec_result = self.model_asr.generate(
                    input=audio_data, 
                    **generate_kwargs
                )[0]
                
                # 添加标点符号
                if self.model_punc is not None and len(rec_result.get("text", "")) > 0:
                    rec_result = self.model_punc.generate(
                        input=rec_result["text"], 
                        **status_dict_punc
                    )[0]
                
                # 返回结果，包含可能的timestamp信息
                result_dict = {
                    "session_id": session_id,
                    "text": rec_result.get("text", ""),
                    "mode": "offline",
                    "is_final": is_final,
                    "timestamp": time.time()
                }
                
                # 如果有timestamp信息(对应C++的us_alphas和us_peaks)
                if "timestamp" in rec_result:
                    result_dict["word_timestamps"] = rec_result["timestamp"]
                
                return result_dict
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: 离线识别错误: {e}")
            
        return {
            "session_id": session_id,
            "text": "",
            "mode": "offline",
            "is_final": True,
            "timestamp": time.time(),
            "status_dict_asr": status_dict_asr,
            "status_dict_punc": status_dict_punc
        }
    
    def vad_detect(self, audio_data: bytes, status_dict_vad: dict) -> tuple:
        """VAD 语音活动检测
        
        返回:
            (speech_start, speech_end, updated_status_dict_vad)
        """
        try:
            segments_result = self.model_vad.generate(
                input=audio_data, 
                **status_dict_vad
            )[0]["value"]
            
            speech_start = -1
            speech_end = -1
            
            if len(segments_result) == 0 or len(segments_result) > 1:
                return speech_start, speech_end, status_dict_vad
                
            if segments_result[0][0] != -1:
                speech_start = segments_result[0][0]
            if segments_result[0][1] != -1:
                speech_end = segments_result[0][1]
                
            return speech_start, speech_end, status_dict_vad
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: VAD 检测错误: {e}")
            return -1, -1, status_dict_vad
        
    def run(self):
        """工作进程主循环"""
        self.load_model()
        logger.info(f"Worker {self.worker_id}: 准备就绪，等待任务...")
        
        while True:
            try:
                # 从任务队列获取任务
                task = self.task_queue.get()
                
                if task is None:  # 退出信号
                    logger.info(f"Worker {self.worker_id}: 收到退出信号")
                    break
                    
                session_id = task['session_id']
                audio_data = task['audio_data']
                task_type = task.get('type', 'online')
                
                # 执行推理
                start_time = time.time()
                
                if task_type == 'online':
                    # 在线流式识别
                    status_dict = task.get('status_dict', {"cache": {}, "is_final": False})
                    result = self.transcribe_online(audio_data, session_id, status_dict)
                elif task_type == 'offline':
                    # 离线精细识别
                    status_dict_asr = task.get('status_dict_asr', {})
                    status_dict_punc = task.get('status_dict_punc', {"cache": {}})
                    # 对应C++的input_finished参数
                    is_final = task.get('is_final', True)
                    result = self.transcribe_offline(audio_data, session_id, status_dict_asr, status_dict_punc, is_final)
                elif task_type == 'vad':
                    # VAD 检测
                    status_dict_vad = task.get('status_dict_vad', {"cache": {}, "is_final": False})
                    speech_start, speech_end, updated_status_dict_vad = self.vad_detect(audio_data, status_dict_vad)
                    result = {
                        "session_id": session_id,
                        "speech_start": speech_start,
                        "speech_end": speech_end,
                        "timestamp": time.time(),
                        "status_dict_vad": updated_status_dict_vad  # 返回更新后的status_dict
                    }
                else:
                    logger.warning(f"Worker {self.worker_id}: 未知任务类型: {task_type}")
                    continue
                
                inference_time = time.time() - start_time
                
                # 将结果放入结果队列
                result['worker_id'] = self.worker_id
                result['task_type'] = task_type
                result['inference_time'] = inference_time
                self.result_queue.put(result)
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: 处理任务时出错: {e}")
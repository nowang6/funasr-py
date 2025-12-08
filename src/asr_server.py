import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from multiprocessing import Process, Queue, Manager
from typing import Dict, Optional
from numba.core.types import none
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

from src.asr_worker import ASRWorker
from src.logger import logger
from src.session import SessionManager
import base64
from funasr import AutoModel



VAD_MODEL_PATH="models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
VAD_CHUNK_SIZE = 256
CLIENT_CHUNK_SIZE = 128
VAD_CHECK_FREQ = VAD_CHUNK_SIZE // CLIENT_CHUNK_SIZE



ASR_MODEL_PATH="models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
asr_model = AutoModel(
    model=ASR_MODEL_PATH,
    disable_pbar=True,
    disable_log=True,
    disable_update=True,
)

ASR_MODEL_ONLINE_PATH="models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
asr_model_online = AutoModel(
    model=ASR_MODEL_ONLINE_PATH,
    disable_pbar=True,
    disable_log=True,
    disable_update=True,
)

PUNC_MODEL_PATH="models/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"

PUNC_MODEL = AutoModel(
        model=PUNC_MODEL_PATH,
        disable_pbar=True,
        disable_log=True,
        disable_update=True
        )

VAD_MODEL = AutoModel(
    model=VAD_MODEL_PATH,
    disable_pbar=True,
    disable_log=True,
    disable_update=True
    )


offline_model = AutoModel(model=ASR_MODEL_PATH, 
                          vad_model=VAD_MODEL_PATH, 
                          punc_model=PUNC_MODEL_PATH)
                          # spk_model="cam++", spk_model_revision="v2.0.2"


def vad_infer(audio_in):
    res = VAD_MODEL.generate(input=audio_in, cache={}, is_final=False, chunk_size=VAD_CHUNK_SIZE)
    value = res[0]["value"]
    return len(value) == 0


def asr_online_infer(audio_in):
    res = asr_model_online.generate(input=audio_in, cache={}, is_final=False, chunk_size=VAD_CHUNK_SIZE)
    return res[0]["text"]

def asr_offline_infer(audio_in):

    res = offline_model.generate(input=f"{model.model_path}/example/asr_example.wav", 
                     batch_size_s=300, 
                     hotword='魔搭')
    res = PUNC_MODEL.generate(input=res)
    return res[0]["text"]


# 全局变量
NUM_WORKERS = 1
task_queue = None
result_queue = None
session_manager = None
workers = []
stats_lock = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    global task_queue, result_queue, session_manager, workers, stats_lock
    
    logger.info("=" * 50)
    logger.info("启动实时语音转写服务")
    logger.info(f"工作进程数量: {NUM_WORKERS}")
    logger.info("=" * 50)
    
    # 创建队列
    task_queue = Queue(maxsize=5000)
    result_queue = Queue(maxsize=5000)
    
    # 创建会话管理器
    manager = Manager()
    stats_lock = manager.Lock()
    session_manager = SessionManager()
    
    
    yield  # 应用运行期间
    
    # 关闭时清理
    logger.info("关闭服务，清理资源...")
    
    # 发送退出信号给所有工作进程
    for _ in range(NUM_WORKERS):
        task_queue.put(None)
    
    # 等待所有工作进程结束
    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    
    logger.info("所有工作进程已关闭")

# ==================== FastAPI 应用 ====================
app = FastAPI(title="实时语音转写服务", lifespan=lifespan)

@app.websocket("/v1/asr")
async def websocket_asr(websocket: WebSocket):
    """WebSocket 实时语音转写接口"""
    await websocket.accept()
    session_id = f"session_{uuid.uuid4().hex}"
    session_manager.create_session(session_id, websocket)
    continue_slience_count = 0
    has_spoken_before = False
    offline_start_frame_id = 0
    online_start_frame_id = 0
    
    while True:
        # 接收客户端数据
        data = await websocket.receive()
        
        # 客户端发送的是JSON格式的消息
        msg_text = data.get('text')
        if not msg_text:
            logger.warning(f"收到非文本消息: {session_id}")
            continue
        
        try:
            msg = json.loads(msg_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {session_id}, 错误: {e}")
            continue
        
        # 提取header信息
        header = msg.get('header', {})
        trace_id = header.get('traceId', '')
        app_id = header.get('appId', '')
        biz_id = header.get('bizId', '')
        status = header.get('status', 0)  # 0=首帧, 1=中间帧, 2=尾帧
        
        # 提取parameter信息
        parameter = msg.get('parameter', {})
        
        # 提取payload中的音频数据
        payload = msg.get('payload', {})
        audio_payload = payload.get('audio', {})
        audio_b64 = audio_payload.get('audio', '')
        
        if not audio_b64:
            logger.warning(f"收到空音频数据: {session_id}")
            continue
        
        # 解码base64音频数据
        try:
            audio_frame = base64.b64decode(audio_b64)
        except Exception as e:
            logger.error(f"音频数据base64解码失败: {session_id}, 错误: {e}")
            continue
        
        logger.debug(f"收到音频数据: session={session_id}, trace_id={trace_id}, "
                    f"status={status}, audio_size={len(audio_frame)}")
        
        # 获取会话对象
        session = session_manager.get_session(session_id)
        if not session:
            logger.error(f"会话不存在: {session_id}")
            break
        
        # 这里只接收数据，不进行推理处理
        # 可以根据需要保存 trace_id, app_id, biz_id 等信息到session中
        session['trace_id'] = trace_id
        session['app_id'] = app_id
        session['biz_id'] = biz_id
        
        if status == 0:
             # 首帧：重置状态，开始新的识别
             session["frames"] = []
             
        session["frames"].append(audio_frame)
        
        frames_count = len(session["frames"])
        
        if frames_count % VAD_CHECK_FREQ == 0:
            ## 每VAD_CHECK_FREQ帧执行一次VAD识别
            vad_accumulate_frame = b"".join(session["frames"][-VAD_CHECK_FREQ:])
            silence = vad_infer(vad_accumulate_frame)
            if silence:
                continue_slience_count += 1
                # 如果当前帧没有人说话，前序帧也没有人说话的累积
                if not has_spoken_before:
                    offline_start_frame_id = frames_count
                    online_start_frame_id = frames_count
                # 如果没有人说话，跳过后续推理
                continue
            else:
                has_spoken_before = True
                continue_slience_count = 0
                
            
        if has_spoken_before and (continue_slience_count == 2 or status==2):
            ## 连续2次静音，或者尾帧，触发离线识别
            offline_accumulate_frame = b"".join(session["frames"][offline_start_frame_id+1:])
            res = asr_offline_infer(offline_accumulate_frame)
            #离线识别完成，重置someone_speak标志, 离线开始frames id
            offline_start_frame_id = frames_count
            has_spoken_before = False
            # 离线识别完成，跳过在线识别
            continue
        if frames_count % 4 == 0:
            ## 每4帧执行一次在线识别
            online_accumulate_frame = b"".join(session["frames"][online_start_frame_id+1:])
            res = asr_online_infer(online_accumulate_frame)
            online_start_frame_id = frames_count
    
if __name__ == "__main__":
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1  # 注意：只能用1个worker，因为我们使用了全局变量
    )
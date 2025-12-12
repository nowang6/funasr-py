import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from multiprocessing import Process, Queue, Manager, cpu_count
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional
from numba.core.types import none
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sympy.core.symbol import Str
import uvicorn

from src.asr_worker import ASRWorker
from src.vad_worker import VADWorker

from src.asr_session import SessionManager
import base64
from src.const import VAD_CHUNK_SIZE, VAD_PROCESS_NUM, NUM_WORKERS, AUDIO_CHUNK_SIZE
from src.logger import logger
from src.async_tasks import audio_handler, cleanup_tasks, vad_result_dispatcher, result_handler, vad_check


# 全局变量
vad_task_queue = None
vad_result_queue = None
task_queue = None
result_queue = None
session_manager = None
workers = []
stats_lock = None
vad_executor = None  # VAD进程池执行器
vad_result_dict = None  # 用于存储VAD结果，key为task_id，value为结果



def worker_process(worker_id: int, task_queue: Queue, result_queue: Queue):
    worker = ASRWorker(worker_id, task_queue, result_queue)
    worker.run()
    
def vad_worker_process(worker_id: int, task_queue: Queue, result_queue: Queue):
    """工作进程入口函数"""
    worker = VADWorker(worker_id, task_queue, result_queue)
    worker.run()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    global vad_task_queue, vad_result_queue, task_queue, result_queue, session_manager, workers, stats_lock, vad_executor, vad_result_dict, vad_result_lock
    
    logger.info("=" * 50)
    logger.info("启动实时语音转写服务")
    logger.info(f"工作进程数量: {NUM_WORKERS}")
    
    # 创建队列
    vad_task_queue = Queue(maxsize=5000)
    vad_result_queue = Queue(maxsize=5000)
    task_queue = Queue(maxsize=5000)
    result_queue = Queue(maxsize=5000)
    
    # 创建会话管理器
    manager = Manager()
    stats_lock = manager.Lock()
    vad_result_dict = manager.dict()  # 用于存储VAD结果
    vad_result_lock = manager.Lock()  # 保护结果字典的锁
    session_manager = SessionManager()
    
    
    # 启动VAD进程
    for i in range(VAD_PROCESS_NUM):
        p = Process(target=vad_worker_process, args=(i, vad_task_queue, vad_result_queue))
        p.start()
        workers.append(p)
        logger.info(f"启动 VAD process {i}")
    
    # 启动工作进程
    for i in range(NUM_WORKERS):
        p = Process(target=worker_process, args=(i, task_queue, result_queue))
        p.start()
        workers.append(p)
        logger.info(f"启动 Worker {i}")
    
    # 启动结果处理任务
    # asyncio.create_task(result_handler())
    
    # 启动VAD结果分发任务
    asyncio.create_task(vad_result_dispatcher(vad_result_queue, vad_result_dict))
    
    # 启动清理任务
    asyncio.create_task(cleanup_tasks(session_manager, vad_result_dict))
    
    # 启动统计日志任务
    asyncio.create_task(audio_handler())
    
    yield  # 应用运行期间
    
    # 关闭时清理
    logger.info("关闭服务，清理资源...")
    
    # 关闭VAD进程池
    if vad_executor:
        vad_executor.shutdown(wait=True)
        logger.info("VAD进程池已关闭")
    
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


@app.websocket("/tuling/ast/v3")
async def websocket_asr(websocket: WebSocket):
    """WebSocket 实时语音转写接口"""
    await websocket.accept()
    session_id = f"session_{uuid.uuid4().hex}"
    session_manager.create_session(session_id, websocket)
    continue_slience_count = 0
    offline_start_frame_id = 0
    online_start_frame_id = 0
    vad_start_frame_id = 0
    has_spoken_before = False
    
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
        status = header.get('status', 0)  # 0=首帧, 1=中间帧, 2=尾帧
        
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
        
        logger.debug(f"收到音频数据: session={session_id}"
                    f"status={status}, audio_size={len(audio_frame)}")
        
        # 获取会话对象
        session = session_manager.get_session(session_id)
        if not session:
            logger.error(f"会话不存在: {session_id}")
            break
        
        if status == 0:
             # 首帧：重置状态
            session["frames"] = bytearray()
            session["frames_count"] = 0
            session["offline_start_frame_id"] = 0
            session["online_start_frame_id"] = 0
            session["vad_start_frame_id"] = 0
            session["results"] = []
            session["frame_end"] = False
        elif status == 1:
            # 中间帧
            None
        elif status == 2:
            # 尾帧
            session["frame_end"] = True
        else:
            logger.error(f"收到非法状态: {status}")
            continue
        
        session["frames"].extend(audio_frame)
        frames = session["frames"]
        frames_count = len(frames)
        session["frames_count"] = frames_count
        
        #是否执行VAD
        if (frames_count - vad_start_frame_id) >= VAD_CHUNK_SIZE:
            vad_end_frame_id = vad_start_frame_id + VAD_CHUNK_SIZE
            vad_accumulate_frame = frames[vad_start_frame_id:vad_end_frame_id]
            logger.info(f"vad识别帧数: {len(vad_accumulate_frame)}")
            vad_audio_data = bytes(vad_accumulate_frame)
            silence = await vad_check(vad_task_queue, vad_audio_data, vad_result_dict)
            logger.info(f"vad识别结果: {silence}")
            # 识别完成，更新起始标记
            vad_start_frame_id = vad_end_frame_id
            if silence:
                continue_slience_count += 1
                # 连续2次silence，并且前序有人说话，触发离线识别
                if has_spoken_before and continue_slience_count >= 2:
                    offline_end_frame_id = vad_end_frame_id
                    offline_audio_data = bytes(frames[offline_start_frame_id:offline_end_frame_id])
                    # 识别完成后，更新起始标记
                    offline_start_frame_id = offline_end_frame_id
                if not has_spoken_before:
                    # 前序没有人说话，跳过识别
                    continue
            else:
                has_spoken_before = True
                continue_slience_count = 0
        #是否触发在线识别
        if (frames_count -  online_start_frame_id) >= AUDIO_CHUNK_SIZE: 
            None       
            
            
            
     
    
if __name__ == "__main__":
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1  # 注意：只能用1个worker，因为我们使用了全局变量
    )
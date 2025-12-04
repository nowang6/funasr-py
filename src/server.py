import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from multiprocessing import Process, Queue, Manager
from typing import Dict, Optional
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

from src.worker import ASRWorker
from src.logger import logger
from src.session import SessionManager
import base64


def worker_process(worker_id: int, task_queue: Queue, result_queue: Queue):
    """工作进程入口函数"""
    worker = ASRWorker(worker_id, task_queue, result_queue)
    worker.run()

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
    
    # 启动工作进程
    for i in range(NUM_WORKERS):
        p = Process(target=worker_process, args=(i, task_queue, result_queue))
        p.start()
        workers.append(p)
        logger.info(f"启动 Worker {i}")
    
    # 启动结果处理任务
    asyncio.create_task(result_handler())
    
    # 启动统计日志任务
    asyncio.create_task(stats_logger())
    
    # 启动会话清理任务
    asyncio.create_task(session_cleanup_task())
    
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

async def result_handler():
    """处理推理结果并发送回客户端"""
    logger.info("结果处理器启动")
    
    while True:
        try:
            # 非阻塞方式检查结果队列
            if not result_queue.empty():
                result = result_queue.get_nowait()
                session_id = result['session_id']
                task_type = result.get('task_type', '')
                
                # 获取对应的 WebSocket 连接
                session = session_manager.get_session(session_id)
                if session:
                    # 更新最后活动时间
                    session['last_activity'] = time.time()
                    
                    try:
                        if task_type == 'vad':
                            # 处理 VAD 检测结果
                            speech_start = result.get('speech_start', -1)
                            speech_end = result.get('speech_end', -1)
                            
                            if speech_start != -1:
                                # 检测到语音开始
                                session['speech_start'] = True
                                duration_ms = len(session['frames'][-1]) // 32 if session['frames'] else 0
                                beg_bias = (session['vad_pre_idx'] - speech_start) // duration_ms if duration_ms > 0 else 0
                                frames_pre = session['frames'][-beg_bias:] if beg_bias > 0 else []
                                session['frames_asr'] = []
                                session['frames_asr'].extend(frames_pre)
                            
                            if speech_end != -1:
                                # 检测到语音结束
                                logger.info(f"[VAD检测到语音结束] session={session_id}, speech_end={speech_end}")
                                session['speech_end_i'] = speech_end
                        
                        elif task_type in ['online', 'offline']:
                            # 处理识别结果（全双工2pass场景）
                            text = result.get('text', '')
                            if len(text) > 0:
                                is_final = result.get('is_final', False)
                                
                                # 确定消息类型和状态：符合客户端test_asr.py的格式
                                if session['mode'] == '2pass':
                                    if task_type == 'online':
                                        # 第一遍在线识别：中间状态(progressive)
                                        msgtype = 'progressive'
                                        status = 1  # 中间帧
                                    elif task_type == 'offline':
                                        # 第二遍离线识别：最终状态(sentence)
                                        msgtype = 'sentence'
                                        status = 2 if is_final else 1  # 最终结果时为尾帧
                                else:
                                    msgtype = 'sentence' if is_final else 'progressive'
                                    status = 2 if is_final else 1
                                
                                # 将文本转换为 ws 格式（词语数组）
                                # 简单处理：将整个文本作为一个词语单元
                                ws_array = []
                                if text:
                                    # 将文本按标点符号或空格分割成词组
                                    import re
                                    # 按标点和空格分割，保留标点
                                    segments = re.split(r'([，。！？、；：,.!?;:\s]+)', text)
                                    for segment in segments:
                                        if segment and segment.strip():
                                            ws_array.append({
                                                "cw": [{
                                                    "w": segment,
                                                    "rl": 0  # 角色ID，0表示无角色
                                                }]
                                            })
                                    
                                    # 如果没有分割结果，至少包含完整文本
                                    if not ws_array:
                                        ws_array.append({
                                            "cw": [{
                                                "w": text,
                                                "rl": 0
                                            }]
                                        })
                                
                                # 计算时间戳（毫秒）
                                current_time_ms = int(time.time() * 1000)
                                bg = session.get('last_bg', 0)
                                ed = current_time_ms - session.get('start_time_ms', current_time_ms)
                                session['last_bg'] = ed  # 更新下次的开始时间
                                
                                # 构建符合客户端test_asr.py格式的响应消息
                                response = {
                                    "header": {
                                        "traceId": session.get('trace_id', ''),
                                        "appId": session.get('app_id', ''),
                                        "bizId": session.get('biz_id', ''),
                                        "status": status,  # 1=中间帧, 2=尾帧
                                        "resIdList": []
                                    },
                                    "payload": {
                                        "result": {
                                            "bg": bg,  # 开始时间（毫秒）
                                            "ed": ed,  # 结束时间（毫秒）
                                            "msgtype": msgtype,  # progressive=中间状态(在线), sentence=最终状态(离线)
                                            "ws": ws_array  # 词语数组
                                        }
                                    }
                                }
                                
                                # 发送结果到客户端
                                await session['websocket'].send_json(response)
                                
                                # 日志记录
                                logger.debug(f"发送识别结果: session={session_id}, "
                                           f"msgtype={msgtype}, text_len={len(text)}, status={status}")
                    except Exception as e:
                        logger.error(f"发送结果失败: {session_id}, 错误: {e}")
                        # 标记session为失败，但不在这里删除，让WebSocket handler的finally块处理
                        session['send_failed'] = True
            else:
                await asyncio.sleep(0.01)  # 避免CPU空转
                
        except Exception as e:
            logger.error(f"结果处理器错误: {e}")
            await asyncio.sleep(0.1)

async def process_audio_data(session_id: str, audio_data: bytes, session: dict):
    """处理音频数据，执行 VAD 检测和在线识别"""
    # 将音频帧加入缓冲
    session['frames'].append(audio_data)
    duration_ms = len(audio_data) // 32  # 假设 16kHz, 16bit, mono
    session['vad_pre_idx'] += duration_ms
    
    # 在线识别：累积音频帧
    session['frames_asr_online'].append(audio_data)
    session['status_dict_asr_online']['is_final'] = session['speech_end_i'] != -1
    
    # 更新 VAD chunk_size
    if 'chunk_size' in session['status_dict_asr_online']:
        session['status_dict_vad']['chunk_size'] = int(
            session['status_dict_asr_online']['chunk_size'][1] * 60 / session['chunk_interval']
        )
    
    # 调试日志：检查条件判断
    frames_count = len(session['frames_asr_online'])
    chunk_interval = session['chunk_interval']
    is_final = session['status_dict_asr_online']['is_final']
    speech_end_i = session['speech_end_i']
    condition1 = frames_count % chunk_interval == 0
    condition2 = is_final
    logger.debug(f"[DEBUG] session={session_id}, frames_count={frames_count}, "
                 f"chunk_interval={chunk_interval}, condition1={condition1}, "
                 f"speech_end_i={speech_end_i}, is_final={is_final}, condition2={condition2}")
    
    # 当累积足够帧数或检测到语音结束时，进行在线识别
    if (condition1 or condition2):
        logger.info(f"[进入在线识别分支] session={session_id}, frames_count={frames_count}, "
                    f"condition1={condition1}, condition2={condition2}")
        audio_in = b"".join(session['frames_asr_online'])
        await asyncio.to_thread(
            task_queue.put,
            {
                'session_id': session_id,
                'audio_data': audio_in,
                'type': 'online',
                'status_dict': session['status_dict_asr_online'].copy()
            }
        )
        session['first_pass_count'] += 1
        session_manager.stats['total_requests'] += 1
        session['frames_asr_online'] = []
    else:
        logger.debug(f"[未进入在线识别分支] session={session_id}, 需要满足任一条件: "
                     f"frames_count({frames_count}) % chunk_interval({chunk_interval}) == 0 或 is_final=True")
    
    # VAD 检测
    if session['speech_start']:
        session['frames_asr'].append(audio_data)
    
    # 执行 VAD 检测（异步提交到队列）
    await asyncio.to_thread(
        task_queue.put,
        {
            'session_id': session_id,
            'audio_data': audio_data,
            'type': 'vad',
            'status_dict_vad': session['status_dict_vad'].copy()
        }
    )
    
    # 如果检测到语音结束，触发离线识别
    if session['speech_end_i'] != -1 or not session['is_speaking']:
        await handle_speech_end(session_id, session)

async def handle_speech_end(session_id: str, session: dict):
    """处理语音结束，执行离线精细识别"""
    
    if len(session['frames_asr']) > 0:
        audio_in = b"".join(session['frames_asr'])
        await asyncio.to_thread(
            task_queue.put,
            {
                'session_id': session_id,
                'audio_data': audio_in,
                'type': 'offline',
                'status_dict_asr': session['status_dict_asr'].copy(),
                'status_dict_punc': session['status_dict_punc'].copy()
            }
        )
        session['second_pass_count'] += 1
        session_manager.stats['total_requests'] += 1

    # 重置状态
    session['frames_asr'] = []
    session['speech_start'] = False
    session['frames_asr_online'] = []
    session['status_dict_asr_online']['cache'] = {}
    
    if not session['is_speaking']:
        session['vad_pre_idx'] = 0
        session['frames'] = []
        session['status_dict_vad']['cache'] = {}
    else:
        # 保留最近的一些帧
        session['frames'] = session['frames'][-20:]
    
    session['speech_end_i'] = -1

async def stats_logger():
    """定期打印统计信息"""
    while True:
        await asyncio.sleep(10)  # 每10秒打印一次
        
        queue_size = task_queue.qsize()
        result_queue_size = result_queue.qsize()
        
        logger.info("=" * 80)
        logger.info(f"【系统统计】时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  活跃会话数: {session_manager.stats['active_sessions']}")
        logger.info(f"  任务队列长度: {queue_size}")
        logger.info(f"  结果队列长度: {result_queue_size}")
        logger.info(f"  总请求数: {session_manager.stats['total_requests']}")
        
        # 检查资源竞争情况
        if queue_size > NUM_WORKERS * 2:
            logger.warning(f"⚠️  任务队列堆积严重 ({queue_size})，工作进程可能不足！")
        
        if session_manager.stats['active_sessions'] > NUM_WORKERS:
            logger.warning(f"⚠️  活跃会话数 ({session_manager.stats['active_sessions']}) 超过工作进程数 ({NUM_WORKERS})，可能出现竞争！")
        
        logger.info("=" * 80)

async def session_cleanup_task():
    """定期清理超时的会话"""
    while True:
        await asyncio.sleep(2*60*60)  # 每2小时检查一次
        try:
            # 清理超时30分钟的会话
            cleaned = session_manager.cleanup_timeout_sessions(timeout_seconds=30*60)
            if cleaned > 0:
                logger.info(f"清理了 {cleaned} 个超时会话")
        except Exception as e:
            logger.error(f"会话清理任务错误: {e}")

@app.websocket("/v1/asr")
async def websocket_asr(websocket: WebSocket):
    """WebSocket 实时语音转写接口"""
    await websocket.accept()
    session_id = f"session_{uuid.uuid4().hex}"
    
    try:
        # 创建会话
        session_manager.create_session(session_id, websocket)
        
        
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
                audio_data = base64.b64decode(audio_b64)
                
            except Exception as e:
                logger.error(f"音频数据base64解码失败: {session_id}, 错误: {e}")
                continue
            
            logger.debug(f"收到音频数据: session={session_id}, trace_id={trace_id}, "
                        f"status={status}, audio_size={len(audio_data)}")
            
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
            
            # 处理音频数据
            await process_audio_data(session_id, audio_data, session)
            
            # 如果是尾帧，标记会话结束
            if status == 2:
                logger.info(f"收到结束帧: {session_id}, trace_id={trace_id}")
                session['is_final'] = True
                break
                    
    except WebSocketDisconnect:
        logger.info(f"客户端断开连接: {session_id}")
    except RuntimeError as e:
        # 捕获 "Cannot call 'receive' once a disconnect message has been received" 错误
        if "disconnect message has been received" in str(e):
            logger.info(f"客户端已断开连接: {session_id}")
        else:
            logger.error(f"WebSocket 运行时错误: {session_id}, {e}")
    except Exception as e:
        logger.error(f"WebSocket 错误: {session_id}, {e}")
    finally:
        # 清理会话资源
        try:
            # 尝试关闭WebSocket连接
            session = session_manager.get_session(session_id)
            if session and session.get('websocket'):
                try:
                    await session['websocket'].close()
                except Exception:
                    pass  # WebSocket可能已经关闭
        except Exception:
            pass
        finally:
            # 移除会话
            session_manager.remove_session(session_id)

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        'status': 'healthy',
        'workers': NUM_WORKERS,
        'active_sessions': session_manager.stats['active_sessions'],
        'task_queue_size': task_queue.qsize(),
        'result_queue_size': result_queue.qsize()
    }

@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    return {
        'active_sessions': session_manager.stats['active_sessions'],
        'total_requests': session_manager.stats['total_requests'],
        'task_queue_size': task_queue.qsize(),
        'result_queue_size': result_queue.qsize(),
        'workers': NUM_WORKERS
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="debug",
        workers=1  # 注意：只能用1个worker，因为我们使用了全局变量
    )
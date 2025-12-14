from src.logger import logger
import asyncio
from multiprocessing import Process, Queue, Manager, cpu_count
from src.asr_session import SessionManager
import time
from src.util import uuid_with_time
async def audio_handler():
    """处理推理结果并发送回客户端"""
    logger.info("结果处理器启动")


async def result_handler(result_queue: Queue):
    """处理推理结果并发送回客户端"""
    logger.info("结果处理器启动")
    while True:
        result = await asyncio.to_thread(result_queue.get)
        if result:
            logger.info(f"收到推理结果: {result}")
        else:
            logger.warning("收到空结果")
            await asyncio.sleep(0.1)


async def cleanup_tasks(session_manager: SessionManager, all_result_dict: tuple):
    while True:
        await asyncio.sleep(12 * 60 * 60)  # 每12小时检查一次
        
        # 清理所有result_dict
        try:
            current_time_ms = int(time.time() * 1000)
            timeout_ms = 30 * 60 * 1000  # 30分钟转换为毫秒
            
            # 遍历所有字典
            for dict_idx, result_dict in enumerate(all_result_dict):
                try:
                    cleaned_count = 0
                    
                    # 获取所有task_id的副本，避免在迭代时修改字典
                    task_ids = list(result_dict.keys())
                    
                    for task_id in task_ids:
                        try:
                            # 从task_id中提取时间戳（格式：timestamp_ms_uuid）
                            if '_' in task_id:
                                timestamp_str = task_id.split('_')[0]
                                timestamp_ms = int(timestamp_str)
                                
                                # 如果超过30分钟，删除该条目
                                if current_time_ms - timestamp_ms > timeout_ms:
                                    result_dict.pop(task_id, None)
                                    cleaned_count += 1
                        except (ValueError, IndexError) as e:
                            # 如果task_id格式不正确，也删除它
                            logger.warning(f"无效的task_id格式，将删除: {task_id}, 错误: {e}")
                            result_dict.pop(task_id, None)
                            cleaned_count += 1
                    
                    if cleaned_count > 0:
                        logger.info(f"清理了字典 {dict_idx} 中的 {cleaned_count} 个超过30分钟的结果条目")
                except Exception as e:
                    logger.error(f"清理字典 {dict_idx} 时发生错误: {e}")
        except Exception as e:
            logger.error(f"结果清理任务错误: {e}")



async def vad_result_dispatcher(queue: Queue, result_dict):
    """VAD结果分发任务：从队列中取出结果并存储到字典中"""
    logger.info("VAD结果分发器启动")
    while True:
        try:
            # 从队列中获取结果（阻塞调用，在线程中执行）
            vad_result = await asyncio.to_thread(queue.get)
            task_id = vad_result.get('task_id')
            if task_id:
                # 将结果存储到字典中
                # 注意：Manager().dict()的单个操作是线程安全的，且asyncio是单线程的
                # 所以不需要锁，也不需要asyncio.to_thread()
                result_dict[task_id] = vad_result
                logger.debug(f"VAD结果已存储: task_id={task_id}")
            else:
                logger.warning(f"收到没有task_id的VAD结果: {vad_result}")
        except Exception as e:
            logger.error(f"VAD结果分发器错误: {e}")
            await asyncio.sleep(0.1)  # 出错时稍作等待

async def asr_result_dispatcher(queue: Queue, result_dict):
    """在线识别结果分发任务：从队列中取出结果并存储到字典中"""
    logger.info("在线识别结果分发器启动")
    while True:
        try:
            online_result = await asyncio.to_thread(queue.get)
            task_id = online_result.get('task_id')
            if task_id:
                result_dict[task_id] = online_result
                logger.debug(f"在线识别结果已存储: task_id={task_id}")
            else:
                logger.warning(f"收到没有task_id的在线识别结果: {online_result}")
        except Exception as e:
            logger.error(f"在线识别结果分发器错误: {e}")
            await asyncio.sleep(0.1)  # 出错时稍作等待

async def vad_check(queue: Queue, audio_data, vad_result_dict: dict) -> bool:
    task_id = uuid_with_time()
    # 发送任务到vad_worker
    await asyncio.to_thread(
        queue.put,
        {
            'task_id': task_id,
            'audio_data': audio_data
        }
    )
    check_interval = 0.01  # 检查间隔10ms
    # 最多检查200次， 2秒
    max_checks = 200
    check_count = 0
    start_time = time.time()
    while True:
        if task_id in vad_result_dict:
            silence = vad_result_dict.pop(task_id)  # 取出并删除
            break
        # 检查超时
        if check_count > max_checks:
            logger.error(f"等待VAD结果超时: task_id={task_id}")
            silence = True  # 超时默认认为是静音
            break
        await asyncio.sleep(check_interval)
        check_count += 1
    return silence

async def online_infer(queue: Queue, audio_data, result_dict: dict) -> bool:
    task_id = uuid_with_time()
    await asyncio.to_thread(
        queue.put,
        {
            'task_id': task_id,
            'audio_data': audio_data
            'type': 'online'
        }
    )
    check_interval = 0.02  # 检查间隔20ms
    # 最多检查200次， 4秒
    max_checks = 200
    check_count = 0
    text = ""
    while True:
        if task_id in result_dict:
            result = result_dict.pop(task_id)
            text = result.get('text', '')
            break
        if check_count > max_checks:
            logger.error(f"等待在线识别结果超时: task_id={task_id}")
            break
        check_count += 1
        await asyncio.sleep(check_interval)
    return text

async def offline_infer(queue: Queue, audio_data, result_dict: dict) -> bool:
    task_id = uuid_with_time()
    await asyncio.to_thread(
        queue.put,
        {
            'task_id': task_id,
            'audio_data': audio_data,
            'type': 'offline'
        }
    )
    check_interval = 0.05  # 检查间隔50ms
    # 最多检查200次， 10秒
    max_checks = 200
    check_count = 0
    text = ""
    while True:
        if task_id in result_dict:
            result = result_dict.pop(task_id)
            text = result.get('text', '')
            break
        if check_count > max_checks:
            logger.error(f"等待离线识别结果超时: task_id={task_id}")
            break
        check_count += 1
        await asyncio.sleep(check_interval)
    return text
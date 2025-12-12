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


async def cleanup_tasks(session_manager: SessionManager, vad_result_dict: dict):
    while True:
        await asyncio.sleep(12 * 60 * 60)  # 每12小时检查一次
        
        # 清理vad_result_dict
        try:
            current_time_ms = int(time.time() * 1000)
            timeout_ms = 30 * 60 * 1000  # 30分钟转换为毫秒
            cleaned_count = 0
            
            # 获取所有task_id的副本，避免在迭代时修改字典
            task_ids = list(vad_result_dict.keys())
            
            for task_id in task_ids:
                try:
                    # 从task_id中提取时间戳（格式：timestamp_ms_uuid）
                    if '_' in task_id:
                        timestamp_str = task_id.split('_')[0]
                        timestamp_ms = int(timestamp_str)
                        
                        # 如果超过30分钟，删除该条目
                        if current_time_ms - timestamp_ms > timeout_ms:
                            vad_result_dict.pop(task_id, None)
                            cleaned_count += 1
                except (ValueError, IndexError) as e:
                    # 如果task_id格式不正确，也删除它
                    logger.warning(f"无效的task_id格式，将删除: {task_id}, 错误: {e}")
                    vad_result_dict.pop(task_id, None)
                    cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"清理了 {cleaned_count} 个超过30分钟的VAD结果条目")
        except Exception as e:
            logger.error(f"VAD结果清理任务错误: {e}")



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
    # 等待处理结果，循环检查结果字典直到找到匹配的task_id
    # 这样可以确保在并发情况下，每个请求都能收到对应的结果
    max_wait_time = 2.0  # 最大等待时间2秒
    check_interval = 0.01  # 检查间隔10ms
    start_time = time.time()
    while True:
        # 直接在协程中操作，不需要锁
        # 原因：1) asyncio是单线程的，不会有并发问题
        #       2) Manager().dict()的单个操作是线程安全的
        #       3) 多个操作的组合在asyncio单线程环境下也是安全的
        if task_id in vad_result_dict:
            vad_result = vad_result_dict.pop(task_id)  # 取出并删除
        else:
            vad_result = None
        if vad_result is not None:
            # 找到匹配的结果
            silence = vad_result['silence']
            break
        # 检查超时
        if time.time() - start_time > max_wait_time:
            logger.error(f"等待VAD结果超时: task_id={task_id}")
            silence = True  # 超时默认认为是静音
            break
        await asyncio.sleep(check_interval)
    return silence
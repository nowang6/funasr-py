
import time
from multiprocessing import Process, Queue, Manager
from funasr import AutoModel

from src.logger import logger
from src.const import ONLINE_MODEL_PATH, VAD_CHUNK_SIZE, VAD_DEFAULT_RESULT


class OnlineWorker:
    def __init__(self, worker_id: int, task_queue: Queue, result_queue: Queue):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue        
        self.model = None

    def load_model(self):
        model_path = ONLINE_MODEL_PATH
        logger.info(f"Worker {self.worker_id}: 开始加载模型: {model_path}")
        self.model = AutoModel(
            model=model_path,
            disable_pbar=True,
            disable_log=True,
            disable_update=True
            )
        logger.info(f"Worker {self.worker_id}: 模型加载完成")
    
   
    def run(self):
        """工作进程主循环"""
        self.load_model()
        logger.info(f"Worker {self.worker_id}: 准备就绪，等待任务...")
        
        while True:
            # 工作进程模式：没有任务时线程会等待，不会消耗 CPU
            task = self.task_queue.get()
            try:
                # 检查退出信号
                if task is None:
                    logger.info(f"Worker {self.worker_id}: 收到退出信号")
                    break
                
                audio_data = task['audio_data']
                res = self.model.generate(input=audio_data, cache={}, is_final=True, chunk_size=VAD_CHUNK_SIZE)
                value = res[0]["value"]
                silence = len(value) == 0
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: 处理任务时出错: {e}")
                silence = VAD_DEFAULT_RESULT
            result = {
                    'task_id': task.get('task_id'),  # 包含task_id用于匹配请求和响应
                    'silence': silence
                }
            self.result_queue.put(result)
               
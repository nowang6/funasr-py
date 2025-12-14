
import time
from multiprocessing import Process, Queue, Manager
from funasr import AutoModel

from src.logger import logger
from src.const import ONLINE_MODEL_PATH, OFFLINE_MODEL_PATH, VAD_MODEL_PATH, PUNC_MODEL_PATH, ONLINE_MODEL_INSTANCE_NUM


class AsrWorker:
    def __init__(self, worker_id: int, task_queue: Queue, result_queue: Queue):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue        
        self.online_models = []  # 存储多个online_model实例
        self.offline_model = None
        self.online_model_index = 0  # 用于轮询选择online_model
    
    def load_models(self):
        # 加载多个online_model实例（按照1:ONLINE_MODEL_INSTANCE_NUM比例）
        for i in range(ONLINE_MODEL_INSTANCE_NUM):
            logger.info(f"Worker {self.worker_id}: 开始加载online模型 {i+1}/{ONLINE_MODEL_INSTANCE_NUM}: {ONLINE_MODEL_PATH}")
            online_model = AutoModel(
                model=ONLINE_MODEL_PATH,
                disable_pbar=True,
                disable_log=True,
                disable_update=True
            )
            self.online_models.append(online_model)
            logger.info(f"Worker {self.worker_id}: online模型 {i+1}/{ONLINE_MODEL_INSTANCE_NUM} 加载完成")
        
        # 加载1个offline_model实例
        logger.info(f"Worker {self.worker_id}: 开始加载offline模型: {OFFLINE_MODEL_PATH}")
        self.offline_model = AutoModel(model=OFFLINE_MODEL_PATH, model_revision="v2.0.4",
                  vad_model=VAD_MODEL_PATH, vad_model_revision="v2.0.4",
                  punc_model=PUNC_MODEL_PATH, punc_model_revision="v2.0.4",
                  #spk_model="cam++", spk_model_revision="v2.0.2",
                  )
        logger.info(f"Worker {self.worker_id}: 模型加载完成 ({ONLINE_MODEL_INSTANCE_NUM}个online + 1个offline)")
    
   
    def run(self):
        """工作进程主循环"""
        self.load_models()
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
                # 轮询使用online_model（3个实例）
                online_model = self.online_models[self.online_model_index]
                self.online_model_index = (self.online_model_index + 1) % len(self.online_models)
                model_out = online_model.generate(input=audio_data, cache={}, is_final=False, chunk_size=[0, 8, 0], encoder_chunk_look_back=0, decoder_chunk_look_back=0)
                text = model_out[0]["text"]
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: 处理任务时出错: {e}")
                text = ""
            result = {
                    'task_id': task.get('task_id'),  # 包含task_id用于匹配请求和响应
                    'text': text
                }
            self.result_queue.put(result)
               
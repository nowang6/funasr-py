# server.py
import uuid
import asyncio
import multiprocessing as mp
from fastapi import FastAPI, WebSocket
from worker import worker_loop
import time

app = FastAPI()

NUM_WORKERS = 8
task_queue = mp.Queue()
result_queue = mp.Queue()
workers = []


# 启动多进程 worker
def start_workers():
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker_loop, args=(task_queue, result_queue, i))
        p.daemon = True
        p.start()
        workers.append(p)


start_workers()


@app.websocket("/tuling/ast/v3")
async def websocket_api(ws: WebSocket):
    await ws.accept()
    
    last_log_time = 0
    LOG_INTERVAL = 5.0  # 每 1 秒打印一次队列长度

    # WebSocket→结果 ID 映射
    pending = {}

    async def result_collector():
        """后台协程，收集 worker 推理结果并发回 websocket."""
        loop = asyncio.get_event_loop()
        while True:
            req_id, output, worker_id = await loop.run_in_executor(
                None, result_queue.get
            )
            websocket = pending.pop(req_id, None)
            if websocket:
                await websocket.send_json({
                    "req_id": req_id,
                    "worker_id": worker_id,
                    "output": output,
                })

    # 启动异步结果监听器
    asyncio.create_task(result_collector())

    while True:
        data = await ws.receive_text()

        # 生成唯一请求 ID
        req_id = str(uuid.uuid4())
        pending[req_id] = ws

        # 派发到 worker
        task_queue.put((req_id, data))
        
        # 打印队大小
        now = time.time()
        if now - last_log_time > LOG_INTERVAL:
            try:
                print("Queue size:", task_queue.qsize())
            except NotImplementedError:
                print("Queue size: not supported on this platform")
            last_log_time = now


# 优雅退出
@app.on_event("shutdown")
def shutdown_event():
    for _ in range(NUM_WORKERS):
        task_queue.put(None)
    for p in workers:
        p.join()

import asyncio
import json
import time
import numpy as np
import websockets

async def simulate_audio_stream(websocket, duration=5):
    """模拟音频流发送"""
    print(f"开始发送音频数据，持续 {duration} 秒...")
    
    # 模拟音频参数：16kHz, 16bit, 单声道
    sample_rate = 16000
    chunk_size = 3200  # 0.2秒的数据
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration:
        # 生成模拟音频数据（实际应用中替换为真实音频）
        audio_chunk = np.random.randint(-32768, 32767, chunk_size, dtype=np.int16)
        audio_bytes = audio_chunk.tobytes()
        
        # 发送音频数据
        await websocket.send(audio_bytes)
        frame_count += 1
        
        # 模拟实时音频流的时间间隔
        await asyncio.sleep(0.2)
        
    print(f"音频发送完成，共发送 {frame_count} 帧")
    
    # 发送语音结束信号
    await websocket.send(json.dumps({
        'type': 'end_of_speech'
    }))
    print("发送语音结束信号")

async def receive_results(websocket):
    """接收识别结果"""
    while True:
        try:
            response = await websocket.recv()
            result = json.loads(response)
            
            if result['type'] == 'connected':
                print(f"✓ 连接成功: {result['session_id']}")
                print(f"  消息: {result['message']}")
                
            elif result['type'] == 'result':
                data = result['data']
                task_type = data.get('task_type', 'unknown')
                symbol = "→" if task_type == 'first_pass' else "⇒"
                
                print(f"\n{symbol} 识别结果 [{task_type}]:")
                print(f"  文本: {data['text']}")
                print(f"  置信度: {data['confidence']:.2%}")
                print(f"  工作进程: Worker {data['worker_id']}")
                print(f"  推理时间: {data['inference_time']:.3f}秒")
                print(f"  是否最终: {data['is_final']}")
                
        except websockets.exceptions.ConnectionClosed:
            print("连接已关闭")
            break
        except Exception as e:
            print(f"接收结果时出错: {e}")
            break

async def test_single_connection():
    """测试单个连接"""
    print("=" * 60)
    print("测试单个WebSocket连接")
    print("=" * 60)
    
    uri = "ws://localhost:8000/v1/asr"
    
    async with websockets.connect(uri) as websocket:
        # 创建接收任务
        receive_task = asyncio.create_task(receive_results(websocket))
        
        # 等待连接确认
        await asyncio.sleep(1)
        
        # 发送音频流
        await simulate_audio_stream(websocket, duration=5)
        
        # 等待一段时间接收剩余结果
        await asyncio.sleep(2)
        
        # 取消接收任务
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass

async def test_concurrent_connections(num_clients=5):
    """测试并发连接"""
    print("=" * 60)
    print(f"测试 {num_clients} 个并发WebSocket连接")
    print("=" * 60)
    
    async def client_worker(client_id):
        uri = "ws://localhost:8000/v1/asr"
        try:
            async with websockets.connect(uri) as websocket:
                print(f"[客户端 {client_id}] 连接成功")
                
                # 接收连接确认
                response = await websocket.recv()
                result = json.loads(response)
                print(f"[客户端 {client_id}] Session ID: {result['session_id']}")
                
                # 创建接收任务
                async def receive():
                    count = 0
                    while count < 10:  # 接收最多10条消息
                        try:
                            resp = await asyncio.wait_for(websocket.recv(), timeout=10)
                            result = json.loads(resp)
                            if result['type'] == 'result':
                                count += 1
                                print(f"[客户端 {client_id}] 收到第 {count} 条结果")
                        except asyncio.TimeoutError:
                            break
                
                receive_task = asyncio.create_task(receive())
                
                # 发送音频
                await simulate_audio_stream(websocket, duration=3)
                
                # 等待接收完成
                await asyncio.wait_for(receive_task, timeout=5)
                
                print(f"[客户端 {client_id}] 测试完成")
                
        except Exception as e:
            print(f"[客户端 {client_id}] 错误: {e}")
    
    # 并发运行多个客户端
    tasks = [client_worker(i) for i in range(num_clients)]
    await asyncio.gather(*tasks)

async def test_heartbeat():
    """测试心跳机制"""
    print("=" * 60)
    print("测试心跳机制")
    print("=" * 60)
    
    uri = "ws://localhost:8000/v1/asr"
    
    async with websockets.connect(uri) as websocket:
        # 接收连接确认
        response = await websocket.recv()
        print(f"连接成功: {json.loads(response)}")
        
        # 发送心跳
        for i in range(3):
            await websocket.send(json.dumps({'type': 'ping'}))
            response = await websocket.recv()
            result = json.loads(response)
            print(f"心跳 {i+1}: {result}")
            await asyncio.sleep(1)

async def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("WebSocket ASR 服务测试")
    print("=" * 60 + "\n")
    
    # 测试1: 单个连接
    await test_single_connection()
    await asyncio.sleep(2)
    
    # 测试2: 并发连接
    await test_concurrent_connections(num_clients=5)
    await asyncio.sleep(2)
    
    # 测试3: 心跳机制
    await test_heartbeat()
    
    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
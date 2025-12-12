import asyncio
import websockets
import base64
import json
import uuid
import os
import wave
from time import time
from datetime import datetime

# 移除代理设置，否则websocket链接报错
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)


WS_URL = "ws://localhost:8000/tuling/ast/v3"
# WS_URL = "ws://125.64.42.236:11924/tuling/ast/v3"
AUDIO_PATH = "data/近远场测试.wav" # 音频文件路径
FRAME_SIZE = 4096
INTERVAL = 0.04
OUTPUT_FILE = "recognition_results.json"  # 输出文件路径

have_msg = False


def gen_trace_id():
    return str(uuid.uuid4())

async def send_audio(ws, audio_path):
    trace_id = gen_trace_id()
    biz_id = "test_bizid_001"
    app_id = "123456"
    status = 0

    # 使用 wave 模块读取 WAV 文件，只读取音频数据（跳过文件头）
    with wave.open(audio_path, "rb") as wav_file:
        # 获取音频参数（可选，用于调试）
        params = wav_file.getparams()
        sample_rate = wav_file.getframerate()
        print(f"音频参数: 采样率={sample_rate}Hz, 声道数={params.nchannels}, "
              f"位深度={params.sampwidth*8}bit, 总帧数={wav_file.getnframes()}")
        
        # 读取所有音频帧数据（不包含 WAV 文件头）
        audio_data = wav_file.readframes(wav_file.getnframes())
        
        # 按 FRAME_SIZE 字节分块发送
        offset = 0
        while offset < len(audio_data):
            # 计算本次读取的数据块大小
            chunk_size = min(FRAME_SIZE, len(audio_data) - offset)
            data = audio_data[offset:offset + chunk_size]
            offset += chunk_size
            
            # 如果读取的数据小于请求的大小，说明这是最后一块数据
            is_last_chunk = chunk_size < FRAME_SIZE or offset >= len(audio_data)

            if is_last_chunk:
                status = 2  # 设置为结束状态
            
            audio_b64 = base64.b64encode(data).decode()
            
            # 构建payload，只在第一块数据时包含text
            payload = {
                "audio": {
                    "audio": audio_b64
                }
            }
            
            # 只在第一块数据时添加热词
            # if status == 0:
            #     payload["text"] = {
            #         "text": "张三疯|向钱看"
            #     }
            
            msg = {
                "header": {
                    "traceId": trace_id,
                    "appId": app_id,
                    "bizId": biz_id,
                    "status": status,
                    "resIdList": []
                },
                "parameter": {
                    "engine": {
                        "wdec_param_LanguageTypeChoice": "5"
                    }
                },
                "payload": payload
            }
            # print(msg)
            await ws.send(json.dumps(msg))
            print(f"发送成功, status: {status}")
        
            
            # 如果是最后一块数据，发送后立即退出循环
            if is_last_chunk or status == 2:
                break
                
            await asyncio.sleep(INTERVAL)
            status = 1  # 仅第一帧 status=0，后续为1

async def receive_result(ws):

    results = []
    sequence_number = 1
    accumulated_text = ""  # 累积的文本内容
    role = "(角色未知)"
    
    async for message in ws:
       
        try:
            resp = json.loads(message)
            
            # 添加序号到结果中
            result_with_sequence = {
                "sequence": sequence_number,
                "timestamp": asyncio.get_event_loop().time(),
                "data": resp
            }
            
            results.append(result_with_sequence)
            
            # 解析识别结果
            if "payload" in resp and "result" in resp["payload"]:
                result = resp["payload"]["result"]
                bg = result.get("bg")
                ed = result.get("ed")
                msgtype = result.get("msgtype", "")
               
                
                # 提取当前帧的文本内容
                current_text = ""
                if "ws" in result:
                    for ws_item in result["ws"]:
                        if "cw" in ws_item:
                            for cw_item in ws_item["cw"]:
                                if "w" in cw_item:
                                    rl = cw_item.get("rl", "")
                                    if rl!=0:
                                        role = f"(角色{rl})"
                                    current_text += (cw_item["w"] + role)
                                    
                                  
                current_text += f"[{bg}-{ed}]"
                # 根据消息类型处理累积文本
                if msgtype == "progressive":
                    # 中间状态：显示当前累积的文本
                    status_label = "【中间状态】"
                    display_text = accumulated_text + current_text
                elif msgtype == "sentence":
                    # 最终状态：将当前文本添加到累积文本中
                    accumulated_text += current_text
                    status_label = "【最终状态】"
                    display_text = accumulated_text
                else:
                    status_label = "【未知状态】"
                    display_text = accumulated_text + current_text
                
                # 打印累积的文本内容
                if current_text or accumulated_text:
                    print(f"收到结果序号#{sequence_number} {status_label}: {display_text}")
                else:
                    print(f"收到结果序号#{sequence_number}: 无文本内容")
            
            # 可以根据 resp["header"]["status"] == 2 判断是否结束
            if resp.get("header", {}).get("status") == 2:
                print("识别结束")
                
                # 保存所有结果到JSON文件
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                print(f"所有识别结果已保存到 {OUTPUT_FILE}")
                print(f"最终累积文本: {accumulated_text}")
                break
                
            sequence_number += 1
            
        except Exception as e:
            print("解析服务端消息失败:", e)

async def main():
    
    # 禁用超时，方便调试服务器端（可以在断点处停留任意长时间）
    async with websockets.connect(
        WS_URL,
        ping_timeout=None,     # 禁用 ping 超时检查
        ping_interval=None,    # 禁用自动 ping（如果需要保持连接，可以设置为较大值如 300）
        close_timeout=120       # 关闭超时时间：1分钟
    ) as ws:
        send_task = asyncio.create_task(send_audio(ws, AUDIO_PATH))
        recv_task = asyncio.create_task(receive_result(ws))
        await asyncio.gather(send_task, recv_task)

if __name__ == "__main__":
    asyncio.run(main())
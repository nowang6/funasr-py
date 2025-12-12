import uuid
import asyncio
import time
from multiprocessing import Queue

def uuid_with_time() -> str:
    return f"{int(time.time() * 1000)}_{uuid.uuid4().hex}"
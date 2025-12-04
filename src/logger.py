import logging

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 关闭 websockets 库的调试日志
logging.getLogger("websockets").setLevel(logging.INFO)
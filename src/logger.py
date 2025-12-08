import logging
import sys

# 创建logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建控制台handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# 创建格式器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 添加handler到logger
logger.addHandler(console_handler)

# 防止日志向上传播导致重复输出
logger.propagate = False

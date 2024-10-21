import logging
import re
from utils.distributed_gpus import is_main_process

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def init_logger(name='result'):
    if is_main_process():
        def format_file_name(text):
            text = text.replace('-', '_')
            text = re.sub(r'[^a-zA-Z0-9_]', '', text).lower()
            return text
        
        name = format_file_name(name)
        file_handler = logging.FileHandler(f'logs/{name}.log')
        file_handler.setLevel(logging.INFO)  # Set the desired log level for this handler
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        return name

def logger(message, level='info'):
    """
    Log messages with different levels: error, info, warning.
    """
    if is_main_process():    
        if level == 'error':
            logging.error(message)
        elif level == 'info':
            logging.info(message)
        elif level == 'warning':
            logging.warning(message)
        else:
            raise ValueError(f"Invalid log level: {level}")
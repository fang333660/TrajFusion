# utils/logger.py (已修改)
import logging
import sys
from copy import copy
from typing import Union, Optional # <<< 确保导入 Optional
import os # <<< 导入 os 以便使用 os.path 和 os.makedirs

from colored import attr, fg

DEBUG = "debug"
INFO = "info"
WARNING = "warning"
ERROR = "error"
CRITICAL = "critical"

LOG_LEVELS = {
    DEBUG: logging.DEBUG,
    INFO: logging.INFO,
    WARNING: logging.WARNING,
    ERROR: logging.ERROR,
    CRITICAL: logging.CRITICAL,
}


class _Formatter(logging.Formatter):
    def __init__(self, colorize=False, *args, **kwargs):
        super(_Formatter, self).__init__(*args, **kwargs)
        self.colorize = colorize

    @staticmethod
    def _process(msg, loglevel, colorize):
        loglevel = str(loglevel).lower()
        if loglevel not in LOG_LEVELS: raise RuntimeError(f"{loglevel} should be one of {LOG_LEVELS}.")
        msg = f"{str(loglevel).upper()}: {str(msg)}"
        if not colorize: return msg
        if loglevel == DEBUG: return "{}{}{}".format(fg(5), msg, attr(0))
        if loglevel == INFO: return "{}{}{}".format(fg(4), msg, attr(0))
        if loglevel == WARNING: return "{}{}{}{}{}".format(fg(214), attr(1), msg, attr(21), attr(0))
        if loglevel == ERROR: return "{}{}{}{}{}".format(fg(202), attr(1), msg, attr(21), attr(0))
        if loglevel == CRITICAL: return "{}{}{}{}{}".format(fg(196), attr(1), msg, attr(21), attr(0))
        return msg  # Fallback

    def format(self, record):
        record = copy(record);
        loglevel = record.levelname
        record.msg = _Formatter._process(str(record.msg), loglevel, self.colorize)
        return super(_Formatter, self).format(record)


class Logger:
    def __init__(self,
                 name: str = "default",
                 colorize: bool = False,
                 log_path: Optional[str] = None,
                 stream=sys.stdout,
                 level: str = INFO,
                 use_file_handler: bool = True):  # <<< 新增 use_file_handler 参数，并给一个默认值
        self.name = name
        self.__logger = logging.getLogger(f"_logger-{name}")
        self.__logger.propagate = False
        self.setLevel(level)
        self.__formatter = _Formatter(
            colorize=colorize,
            fmt="[%(process)d][%(asctime)s.%(msecs)03d @ %(module)s:%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
        )
        self.clear_handlers()
        self.__main_handler = self._add_stream_handler(stream)  # Renamed for clarity

        # ---vvv 只有当 log_path 提供且 use_file_handler 为 True 时才添加文件处理器 vvv---
        if log_path and use_file_handler:  # <<< 使用 use_file_handler
            try:
                log_dir = os.path.dirname(log_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                fh = logging.FileHandler(log_path, "w", encoding='utf-8')
                fh.setFormatter(self.__formatter)
                self.__logger.addHandler(fh)
            except Exception as e:
                print(f"警告: 创建文件日志处理器失败 for path '{log_path}': {e}", file=sys.stderr)
        # ---^^^ ---

        self.debug = self.__logger.debug;
        self.info = self.__logger.info
        self.warning = self.__logger.warning;
        self.error = self.__logger.error
        self.critical = self.__logger.critical

    def _add_stream_handler(self, stream) -> logging.StreamHandler:  # Renamed from add_handler
        handler = logging.StreamHandler(stream)
        handler.setFormatter(self.__formatter)
        self.__logger.addHandler(handler)
        # self.__stream_to_handler[stream] = handler # This dict might not be needed if only one stream handler
        return handler

    def setLevel(self, level: Union[str, int]) -> None:  # (保持不变)
        if isinstance(level, int):
            self.__logger.setLevel(level)
        else:
            level_lower = level.lower()
            if level_lower not in LOG_LEVELS: raise ValueError(f"level should be one of {LOG_LEVELS.keys()}")
            self.__logger.setLevel(LOG_LEVELS[level_lower])

    def clear_handlers(self) -> None:  # (保持不变)
        for handler in self.inner_logger.handlers[:]: self.inner_logger.removeHandler(handler)
        # self.__stream_to_handler = {}

    @property
    def inner_logger(self): return self.__logger
    @property
    def inner_stream_handler(self): return self.__main_handler # This might be an issue if __main_handler isn't always set
    @property
    def inner_formatter(self): return self.__formatter


def log_info(args, logger): # 这个函数现在可能需要从 config 对象而不是 args 对象获取信息
    '''
    output the information about model
    '''
    logger.info('***********************************')
    if hasattr(args, 'data') and hasattr(args.data, 'dataset'):
        logger.info("Dataset: {}".format(args.data.dataset))
    if hasattr(args, 'data') and hasattr(args.data, 'traj_length'):
        logger.info("Trajectory Length: {}".format(args.data.traj_length))
    # ... (以此类推，确保从正确的 config 对象中获取属性) ...
    # 例如:
    if hasattr(args, 'model') and hasattr(args.model, 'guidance_scale'):
        logger.info("Guidance scale (model config): {}".format(args.model.guidance_scale))
    if hasattr(args, 'diffusion') and hasattr(args.diffusion, 'num_diffusion_timesteps'):
        logger.info("Number of steps: {}".format(args.diffusion.num_diffusion_timesteps))
    # ...
    logger.info('***********************************')
    return
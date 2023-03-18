import os
import re
import logging
import datetime

try:
    import codecs
except ImportError:
    codecs = None


class MultiprocessHandler(logging.FileHandler):
    """A file handler for logging to a file.
    """
    def __init__(self, filename, when='D', backupCount=0, encoding=None, delay=False):
        """Initialize the MultiprocessHandler.

        Args:
            filename (str): The file name to write logs to.
            when (str): What to do with the log file.
                        When when='D', the log file will be rotated daily.
                        When when='H', the log file will be rotated hourly.
                        When when='M', the log file will be rotated minute.
                        When when='S', the log file will be rotated second.
            backupCount (int): How many old log files to keep. If backupCount is set to zero, keep all old logs.
            encoding (str): The file encoding.
            delay (bool): Whether to delay the log file rotation. 是否开启 OutSteam缓存。
                          True 表示开启缓存，OutStream输出到缓存，待缓存区满后，刷新缓存区，并输出缓存数据到文件。
                          False表示不缓存，OutStrea直接输出到文件
        """
        self.prefix = filename
        self.when = when.upper()
        self.backupCount = backupCount

        self.extMath = r"^\d{4}-\d{2}-\d{2}"                            # 正则匹配 年-月-日
        self.when_dict = {
            'S': "%Y-%m-%d-%H-%M-%S",
            'M': "%Y-%m-%d-%H-%M",
            'H': "%Y-%m-%d-%H",
            'D': "%Y-%m-%d"
        }
        # 日志文件日期后缀
        self.suffix = self.when_dict.get(when)
        if not self.suffix:
            raise ValueError(u"指定的日期间隔单位无效: %s" % self.when)

        # 拼接文件路径 格式化字符串
        self.filefmt = os.path.join("%s/%s.log" % (self.prefix, self.suffix))

        # 使用当前时间，格式化文件格式化字符串
        self.filePath = datetime.datetime.now().strftime(self.filefmt)

        # 获得文件夹路径
        _dir = os.path.dirname(self.filefmt)
        try:
            # 如果日志文件夹不存在，则创建文件夹
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        except Exception:
            print(u"[INFO] 创建日志文件夹失败")
            print(u"[INFO] 日志文件夹路径：" + self.filePath)

        if codecs is None:
            encoding = None
        logging.FileHandler.__init__(self,self.filePath, 'a+', encoding, delay)

    def shouldChangeFileToWrite(self):
        """是否进行日志切分以更改日志写入的文件

        Returns:
            bool: True 表示需要更新，False 表示不需要更新
        """
        # 以当前时间获得新日志文件路径
        _filePath = datetime.datetime.now().strftime(self.filefmt)
        # 新日志文件日期不等于旧日志文件日期，则表示已经到了日志切分的时候更换日志写入目的为新日志文件。
        # 例如 按天（D）来切分日志：
        #     当前新日志日期等于旧日志日期，则表示在同一天内，还不到日志切分的时候
        #     当前新日志日期不等于旧日志日期，则表示不在同一天内，需要进行日志切分，
        #     将日志内容写入新日志内。
        if _filePath != self.filePath:
            self.filePath = _filePath
            return True
        return False

    def doChangeFile(self):
        """输出信息到日志文件，并删除多于保留个数的所有日志文件

        """
        # 日志文件的绝对路径
        self.baseFilename = os.path.abspath(self.filePath)
        # stream == OutStream
        # stream is not None 表示 OutStream中还有未输出完的缓存数据
        if self.stream:
            # flush close 都会刷新缓冲区，flush不会关闭stream，close则关闭stream
            # self.stream.flush()
            self.stream.close()
            # 关闭stream后必须重新设置stream为None，否则会造成对已关闭文件进行IO操作。
            self.stream = None
        # delay 为False 表示 不OutStream不缓存数据 直接输出所有，只需要关闭OutStream即可
        if not self.delay:
            # 这个地方如果关闭colse那么就会造成进程往已关闭的文件中写数据，从而造成IO错误
            # delay == False 表示的就是 不缓存直接写入磁盘
            # 我们需要重新在打开一次stream
            # self.stream.close()
            self.stream = self._open()
        # 删除多于保留个数的所有日志文件
        if self.backupCount > 0:
            print('[INFO] 删除日志')
            for s in self.getFilesToDelete():
                print(f'[INFO] 删除文件 {s}')
                os.remove(s)

    def getFilesToDelete(self):
        """获得过期需要删除的日志文件
        
        """
        # 分离出日志文件夹绝对路径
        dirName, _ = os.path.split(self.baseFilename)
        fileNames = os.listdir(dirName)
        result = []
        # self.prefix 为日志文件名 列如：mylog.2017-03-19 中的 mylog
        # 加上 点号 . 方便获取点号后面的日期
        prefix = os.path.split(self.prefix)[-1] + '.'
        plen = len(prefix)
        for fileName in fileNames:
            if fileName[:plen] == prefix:
                # 日期后缀 mylog.2017-03-19 中的 2017-03-19
                suffix = fileName[plen:]
                # 匹配符合规则的日志文件，添加到 result 列表中
                if re.compile(self.extMath).match(suffix):
                    result.append(os.path.join(dirName,fileName))
        result.sort()

        # 返回待删除的日志文件
        # 删除多于保留文件个数 backupCount 的所有前面的日志文件。
        if len(result) < self.backupCount:
            result = []
        else:
            result = result[:len(result) - self.backupCount]
        return result

    def emit(self, record):
        """发送一个日志记录
        覆盖FileHandler中的emit方法，logging会自动调用此方法
        """
        try:
            if self.shouldChangeFileToWrite():
                self.doChangeFile()
            logging.FileHandler.emit(self, record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def setup_log(log_file=None, task=''):
    """ logging 使用说明：
        在整个项目中任意想打印日志的地方输入 -> logging.info('I am a logger.')
        各级别的日志会自动保存至 -> log_path

    Args:
        log_file (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # 控制台日志设置
    LOGGING_MSG_FORMAT = '[%(asctime)s] [%(levelname)s] [%(module)s] [%(lineno)d] %(message)s'
    LOGGING_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOGGING_MSG_FORMAT, datefmt=LOGGING_DATE_FORMAT))

    # 创建日志文件夹
    if not log_file:
        log_dir = os.path.abspath('./logs')
    elif os.path.isdir(log_file) or not os.path.splitext(log_file)[1]:
        if not os.path.exists(log_file):
            os.mkdir(log_file)
        log_dir = log_file if 'log' in log_file else os.path.join(log_file, 'logs')
    else:
        log_dir = os.path.basename(log_file)

    if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    # INFO 级别日志输出设置
    info_file_handler = MultiprocessHandler(log_dir, when="D", backupCount=30, encoding='utf-8')

    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(logging.Formatter(LOGGING_MSG_FORMAT, datefmt=LOGGING_DATE_FORMAT))

    # logging.basicConfig(level=logging.INFO, handlers=[console, info_file_handler])
    logging.basicConfig(level=logging.INFO, handlers=[info_file_handler])
    
    return logging


def write_tblog(tblogger, epoch, results, losses):
    """Display mAP and loss information to log."""
    tblogger.add_scalar("val/mAP@0.5", results[0], epoch + 1)
    tblogger.add_scalar("val/mAP@0.50:0.95", results[1], epoch + 1)

    tblogger.add_scalar("train/iou_loss", losses[0], epoch + 1)
    tblogger.add_scalar("train/dist_focalloss", losses[1], epoch + 1)
    tblogger.add_scalar("train/cls_loss", losses[2], epoch + 1)

    tblogger.add_scalar("x/lr0", results[2], epoch + 1)
    tblogger.add_scalar("x/lr1", results[3], epoch + 1)
    tblogger.add_scalar("x/lr2", results[4], epoch + 1)


if __name__ == '__main__':

    logging = setup_log()
    logging.debug('this is a logger debug message')
    logging.info('this is a logger info message')
    logging.warning('this is a logger warning message')
    logging.error('this is a logger error message')

import os
import logging

class Logger:
    def __init__(self, filename='log.txt'):
         # 检查输出文件所在的文件夹是否存在，如果不存在则创建它
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        self.logger = logging.getLogger('custom_logger')
        self.logger.setLevel(logging.INFO)
        self.handler = logging.FileHandler(filename)
        self.handler.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)


    def info(self, message, format=None):
        if format:
            self.handler.setFormatter(logging.Formatter(format))
        self.logger.info(message)
        if format:
            self.handler.setFormatter(self.formatter)


    def error(self, message, format=None):
        if format:
            self.handler.setFormatter(logging.Formatter(format))
        self.logger.error(message)
        if format:
            self.handler.setFormatter(self.formatter)

    
    def warning(self, message, format=None):
        if format:
            self.handler.setFormatter(logging.Formatter(format))
        self.logger.warning(message)
        if format:
            self.handler.setFormatter(self.formatter)


    def critical(self, message, format=None):
        if format:
            self.handler.setFormatter(logging.Formatter(format))
        self.logger.critical(message)
        if format:
            self.handler.setFormatter(self.formatter)

            
    def close(self):
        # self.logger.info('\n\n\n')
        # self.handler.close()
        logging.shutdown()

    def __del__(self):
        # self.info('end...\n\n\n', '%(message)s')
        self.close()

# # 使用示例
# logger = Logger('log/my_log.log')

# logger.info('This is an info log message.')
# logger.error('This is an error log message.', '%(levelname)s - %(message)s')
# logger.warning('This is a warning log message.')
# logger.critical('This is a critical log message.\n\n\n')


# 程序结束时会自动关闭文件?


'''
定义一个Logger类, 实现: 
1. 调用logger=Logger()时能指定输出到那个文件
2. 如果输日志文件在一个文件夹内，则判断该文件见是否存在，若不存在则创建它
3. 调用logger.info(message, format)时能将信息输出到文件并指定输出格式
4. 同样方式定义logger.error等
'''
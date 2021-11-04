import logging
import logging.config
import time

def lazy_logger() -> logging.Logger:
  # date_format = '%Y-%M-%d %I:%M:%S %p'
  # fmt = '[%(asctime)s] %(filename)s.%(funcName)s %(message)s'

  logging.Formatter.converter = time.gmtime
  logging.config.fileConfig('./config/lazy_logger_basic_log.conf')
  logger = logging.getLogger(__name__)

  # quick test
  for i in range(10):
    if i == 0:
      logger.warning('%s - write to stdout', 'WARNING')
    if i == 1:
      logger.debug('%s - write to file', 'DEBUG')

  return logger

lazy_logger()



'''
Brute Force approach

# # Create Handler
  # file_handler = logging.FileHandler( '{0}.log'.format(__file__), mode='a', encoding='utf-8',  delay= True)
  # file_handler.setLevel(logging.DEBUG)
  # file_handler.setFormatter( logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s') )
  # stream_handler = logging.StreamHandler(stream=None) # stderr
  # stream_handler.setLevel(logging.WARNING)
  # stream_handler.setFormatter( logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s') )
  # # Add handlers to logger
  # logger.addHandler(file_handler)
  # logger.addHandler(stream_handler)


'''
import logging
def get_logger(name, file_name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # The following two lines prevent duplicate log output in Jupyter notebook
    if logger.root.handlers:
        logger.root.handlers[0].setLevel(logging.WARNING)

    # Stream handler, sends messages to terminal
    handler_stdout = logging.StreamHandler()
    # Set logger threshold to only output logs with level greater than or equal to INFO
    handler_stdout.setLevel(logging.INFO)
    handler_stdout.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler_stdout)

    # File handler, writes messages to logfile, here the level is set to the lowest debug level
    handler_file = logging.FileHandler(filename=file_name, mode='w', encoding='utf-8')
    # Set to debug level, which means all messages can be written to the log
    handler_file.setLevel(logging.DEBUG)
    handler_file.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler_file)
 
    return logger
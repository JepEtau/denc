import logging

# logging.basicConfig(level=logging.CRITICAL, format='%(filename)s (%(lineno)d): %(funcName)s: %(message)s')
denc_logger: logging.Logger = logging.getLogger("denc_logger")

# Enable this to debug imports
import sys
denc_logger.addHandler(logging.StreamHandler(sys.stdout))
denc_logger.setLevel("ERROR")

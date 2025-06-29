import os
import shutil

from datetime import datetime
from pytz import timezone


def get_datetime(timezone_name='Asia/Seoul'):
    datetime_out = datetime.now(timezone(timezone_name)).strftime('%Y-%m-%d_%H:%M:%S')
    return datetime_out


def init_dirs(dir_list):
    for dir in dir_list:
        if os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
#!/usr/bin/env python3
# @File    : test.py
# @Time    : 9/9/2020 5:14 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm
import threading
import time
from queue import Queue
from threading import Thread
import os
import sys

print(__file__)
print(sys.path[0])
print(os.path.dirname(sys.path[0]))
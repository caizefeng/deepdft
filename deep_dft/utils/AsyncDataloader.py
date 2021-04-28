#!/usr/bin/env python3
# @File    : AsyncDataloader.py
# @Time    : 11/16/2020 9:40 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com

import random
from queue import Queue
# from multiprocessing import JoinableQueue
from threading import Thread

import numpy as np
import torch


class AsyncIterator:
    """Base class for asynchronous loading from iterator"""

    def __init__(self, queue_size, timeout):
        self.queue_size = queue_size
        self.queue = Queue(maxsize=self.queue_size)
        # self.worker = Process(target=self.load_loop)
        self.worker = Thread(target=self.load_loop)
        # self.worker.daemon = True
        self.worker.setDaemon(True)
        self.loader = NotImplemented
        self.timeout = timeout
        self.idx = 0

    def load_loop(self):
        # The loop that will load data into the queue in the background
        return NotImplemented

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        # when the thread is down, if the queue is not empty, do not stop the iteration
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration

        # when a epoch is fully loaded
        elif self.idx >= len(self.loader):
            self.idx = 0
            raise StopIteration

        # otherwise next batch/sample
        else:
            out = self.queue.get(timeout=self.timeout)
            self.queue.task_done()
            self.idx += 1
        return out

    def __len__(self):
        return len(self.loader)


class CudaDataLoader(AsyncIterator):
    """
    Using threads to asynchronously load data from CPU to GPU
    (Especially for those "bigdata" events where neither training set nor test set
    cannot be preloaded to GPU all at once)
    """

    def __init__(self, loader, device, queue_size=2, timeout=None, repeat=False):
        super(CudaDataLoader, self).__init__(queue_size, timeout)
        self.repeat = repeat
        self.device = device
        self.loader = loader
        self.load_stream = torch.cuda.Stream(device=device)
        self.worker.start()

    def load_loop(self):
        if self.repeat:
            while True:
                for i, sample in enumerate(self.loader):
                    # a = time.time()
                    self.queue.put(self.load_instance(sample))
                    # print("test", self.queue.qsize(), time.time() - a)
        else:
            for i, sample in enumerate(self.loader):
                # b = time.time()
                self.queue.put(self.load_instance(sample))
                # print("train", self.queue.qsize(), time.time() - b)

    def load_instance(self, sample):

        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                return sample.to(self.device, non_blocking=True)

        elif sample is None or type(sample) == str:
            return sample
        elif isinstance(sample, dict):
            return {k: self.load_instance(v) for k, v in sample.items()}
        else:
            # when sample is a tuple or a list
            return [self.load_instance(s) for s in sample]

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class FileDataLoader(AsyncIterator):
    """Using threads to asynchronously load npy files into RAM"""

    def __init__(self, path_list, shuffle=True, queue_size=2, timeout=None):
        super(FileDataLoader, self).__init__(queue_size, timeout)
        self.loader = path_list
        self.shuffle = shuffle
        self.worker.start()

    def load_loop(self):
        while True:
            if self.shuffle:
                random.shuffle(self.loader)
            for i, file in enumerate(self.loader):
                self.queue.put(self.load_instance(file))

    def load_instance(self, file):
        if type(file) == str:
            return torch.from_numpy(np.load(file)).float()
        else:
            return [torch.from_numpy(np.load(i)).float() for i in file]

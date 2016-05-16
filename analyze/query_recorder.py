__author__ = 'shengjia'

import os

class Recorder:
    def __init__(self, database_file='query_result'):
        self.database = open(database_file, 'a')

    def ask(self, index):
        self.database.write(str(index) + " ")

    def answer(self, num_fit):
        self.database.write(str(num_fit) + '\n')

    def close(self):
        self.database.close()

if __name__ == '__main__':
    for i in range(0, 10):
        recorder = Recorder()
        recorder.ask(1)
        recorder.answer(4)
        recorder.close()


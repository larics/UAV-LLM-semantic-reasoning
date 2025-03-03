import time
import numpy as np

class Timer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        print(f"Execution time: {self.elapsed_time:.6f} seconds")


class TrajTimer:
    def __init__(self, cf):
        self.cf = cf

    def sleep_while_moving(self):

        prev = np.array(self.cf.get_position())
        time.sleep(0.8)
        curr = np.array(self.cf.get_position())
        if np.linalg.norm(prev - curr) > 0.1:
            self.sleep_while_moving()
        else:
            return
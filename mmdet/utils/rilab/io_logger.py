import numpy as np
import torch
import random


np.set_printoptions(precision=4, suppress=True)


class IOLogger:
    STEP_INDEX = -1
    STEP_LIMIT = 20

    def __init__(self, call_stack='', count_step=False):
        self.call_stack = call_stack
        self.count_step = count_step
        self.file = '/home/falcon/shin_workspace/Datacleaning/log/io_log'
        if count_step:
            with open(self.file, 'w') as f:
                f.write('')

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if self.count_step:
                IOLogger.STEP_INDEX += 1
                print("\n\n#################### STEP:", IOLogger.STEP_INDEX)
                self.fprint(f"\n\n#################### STEP: {IOLogger.STEP_INDEX}")

            self.fprint(f'\n========== ({IOLogger.STEP_INDEX})[{self.call_stack}.{func.__name__}] keyword args')
            # self.print_structure('', kwargs)

            ret = func(*args, **kwargs)

            self.fprint(f'\n========== ({IOLogger.STEP_INDEX})[{self.call_stack}.{func.__name__}] outputs')
            # self.print_structure('', ret)
            return ret
        return wrapper

    def log_var(self, title, var):
        self.fprint(f'\n========== ({IOLogger.STEP_INDEX})[{self.call_stack}] variable: {title}')
        self.print_structure(title, var)

    def fprint(self, *text):
        if IOLogger.STEP_INDEX >= IOLogger.STEP_LIMIT:
            return

        with open(self.file, 'a') as f:
            for t in text:
                if isinstance(t, str):
                    f.write(t)
                else:
                    f.write(f'{t}')
                f.write('\n')

    def print_structure(self, title, data, key=""):
        if isinstance(data, list) or isinstance(data, tuple):
            datalen = len(data)
            for i, datum in enumerate(data[:3]):
                self.print_structure(title, datum, f"{key}/{i}:{datalen}")
        elif isinstance(data, dict):
            for subkey, datum in data.items():
                self.print_structure(title, datum, f"{key}/{subkey}")
        elif type(data) == np.ndarray:
            if data.dtype == bool:
                self.fprint(f'@{title}({key}) shape={data.shape}, type={data.dtype}')
            else:
                self.fprint(f'@{title}({key}) shape={data.shape}, type={data.dtype}, quant={np.quantile(data, np.linspace(0, 1, 6))}')
            sample = data
            for i in range(data.ndim):
                if sample.size < 100:
                    self.fprint(f"\tsample: {'':<10}{sample}")
                    break
                sample = sample[0]
        elif torch.is_tensor(data):
            self.print_structure(title, data.detach().cpu().numpy(), key+'(t)')
        else:
            self.fprint(f'@{title}({key})', data)



class Example:
    @IOLogger(call_stack='Example', count_step=True)
    def init(self, foo, bar, qux):
        return foo

    @IOLogger(call_stack='Example')
    def run(self, foo, bar, qux):
        IOLogger(call_stack='Example').log_var('bar', foo)
        return bar


def main():
    foo = {'np': [np.random.rand(10, 20, 30) for i in range(5)],
           'tensor':[torch.randint(1, 10, (2,3,4)) for i in range(5)],
           'int': [i for i in range(3)],
           'float': [random.random() for i in range(3)]}
    bar = {'fruit': {'apple': [np.random.rand(5, 10, 20) for i in range(5)],
                     'banana': torch.rand((3,8,10), dtype=torch.float32),
                     'mango': 'hellohello'
                     },
           'animal': {'falcon': [np.random.rand(5, 10, 20) for i in range(5)],
                      'bear': torch.rand((3, 8, 10), dtype=torch.float32),
                      'gorilla': 'hellohello'
                     }
           }
    qux = [[torch.randint(1, 10, (2,3,4)) for i in range(5)]]
    exam = Example()
    exam.init(foo, bar=bar, qux=qux)
    exam.run(foo, bar=bar, qux=qux)
    exam.run(foo, bar=bar, qux=qux)
    exam.run(foo, bar=bar, qux=qux)

    exam.init(foo, bar=bar, qux=qux)
    exam.run(foo, bar=bar, qux=qux)

    exam.init(foo, bar=bar, qux=qux)
    exam.run(foo, bar=bar, qux=qux)


if __name__ == '__main__':
    main()



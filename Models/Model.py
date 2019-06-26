from contextlib import redirect_stdout
import os


def save_summary(model, path):
    open(os.path.join(path, 'summary.txt'), 'w')
    with open(path + '/' + 'summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

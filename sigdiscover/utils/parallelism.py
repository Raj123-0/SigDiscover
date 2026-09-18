import multiprocessing as mp
from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar('T')
R = TypeVar('R')

def run_parallel(func: Callable[[T], R], items: Iterable[T], n_jobs: int = -1) -> list[R]:
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    if n_jobs == 1:
        return [func(item) for item in items]

    with mp.Pool(processes=n_jobs) as pool:
        return pool.map(func, items)
import typing as t
from numpy.random import RandomState  # type: ignore


def fork_random_state(rng):
    return RandomState(rng.randint(2**32 - 1))


def tabularize(
        header: t.List[str],
        formats: t.List[str],
        data: t.Iterable[list],
) -> str:
    columns = [[str(h)] for h in header]
    for row in data:
        for col, f, d in zip(columns, formats, row):
            col.append(f.format(d))
    col_size = [max(len(d) for d in col) for col in columns]
    out = []
    out.append(' '.join('-' * size for size in col_size))
    for i in range(len(columns[0])):
        out.append(' '.join(
            col[i].rjust(size) for col, size in zip(columns, col_size)))
    out[0], out[1] = out[1], out[0]
    return '\n'.join(out)

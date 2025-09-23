def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp + 1e-10)


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn + 1e-10)

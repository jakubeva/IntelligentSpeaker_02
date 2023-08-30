import numpy as np
import math
import collections

NEG_INF = -float("inf")  # 表示正无穷


def make_new_beam():
    fn = lambda: (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)


def logsumexp(*args):
    """
    Stable log sum exp.
    解决计算Softmax时出现的上溢或下溢问题
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def decode(probs, beam_size=10, blank=1427):
    """
    将概率序列转换为最佳路径
    :param probs: 一个二维数组，表示每个时间步骤和每个状态的概率
    :param beam_size: 束搜索的宽度，即保留的候选路径数量
    :param blank: 一个特殊状态，表示空白符
    :return: 得分最高的路径和其对数概率的负数
    """
    T, S = probs.shape
    # probs = np.log(probs + 1e-5)
    # 初始化一个束（beam），其中每个候选路径都是一个空序列，初始得分为0
    beam = [(tuple(), (0.0, NEG_INF))]

    # 对于每个时间步骤和每个状态，计算该状态的概率
    for t in range(T):
        next_beam = make_new_beam()
        for s in range(S):
            p = probs[t, s]
            # 对于每个候选路径，根据当前状态的概率更新得分
            for prefix, (p_b, p_nb) in beam:
                if s == blank:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)
                    continue
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s != end_t:
                    n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:
                    n_p_nb = logsumexp(n_p_nb, p_b + p)
                next_beam[n_prefix] = (n_p_b, n_p_nb)
                if s == end_t:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = logsumexp(n_p_nb, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)

        # 根据得分对候选路径进行排序，并保留得分最高的beam_size个路径
        beam = sorted(next_beam.items(), key=lambda x: logsumexp(*x[1]), reverse=True)
        beam = beam[:beam_size]

    best = beam[0]
    # 返回得分最高的路径和其对数概率的负数
    return best[0], -logsumexp(*best[1])


if __name__ == "__main__":
    np.random.seed(3)

    time = 200
    output_dim = 1428

    probs = np.random.rand(time, output_dim)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    labels, score = decode(probs)
    print("labels:%s score:%.3f" % (labels, score))

from puyolib.robustify import robustClassify
import pickle


def deduceMoveSequence(board_seq, nextpuyo_seq):
    """Given a board sequence and the corresponding next puyos, deduce the
    move sequence (including garbage and hidden row usage).

    Also return the revised board sequence with the hidden row included.
    """

    return None


def main():
    record = pickle.load(open("results/testing_results/0:00:09.p", "rb"))
    p1board_seq, p1nextpuyo_seq = robustClassify(record.p1clf)
    p2board_seq, p2nextpuyo_seq = robustClassify(record.p2clf)


if __name__ == "__main__":
    main()

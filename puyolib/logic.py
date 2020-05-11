from collections import namedtuple
from warnings import warn
from puyolib.puyo import Puyo
import numpy as np

from puyolib.robustify import robustClassify
import pickle

"""
The board is 13 rows (12 plus the hidden row) and 6 columns, 0 indexed.
A puyo placement is a puyo type at a board position. The position may be
in row 14 (the vanish row). Only one puyo can be vanished per move.
A puyo move is two puyo placements (one for each of the next puyo pair).
A garbage fall is a list of garbage puyo placements.
A play sequence is a list of puyo moves combined with garbage falls.
"""
PuyoPlacement = namedtuple("PuyoPlacement", ["puyo_type", "row", "col"])
PuyoMove = namedtuple("PuyoMove", ["p1", "p2"])
GarbageFall = namedtuple("GarbageFall", ["placement_list"])
PlaySequence = namedtuple("PlaySequence", ["board", "event_list"])


# In order for hidden and vanish row placements to be deduced, multiple
# board states will be carried forward in parallel wherever there is
# ambiguity. If at the end of the game there remains multiple valid
# play sequences, the play sequence which vanishes the least number
# of puyos will be selected. In case of a tie, the choice will be random.
def deducePlaySequence(board_seq, nextpuyo_seq):
    """Given a board sequence and the corresponding next puyos, deduce the
    move sequence (including garbage and hidden row usage).

    Also return the revised board sequence with the hidden row included.
    """

    # Initialize the starting play sequence.
    board = np.empty(shape=(13, 6), dtype=Puyo)
    board.fill(Puyo.NONE)
    play_sequences = [PlaySequence(board, event_list=[])]

    # Loop through the board sequence to determine valid play sequences.
    nextpuyo_idx = 0
    board_seq = board_seq[1:]
    for board in board_seq:
        for play_sequence in play_sequences:
            deltas = boardDeltas(board, play_sequence.board)

    return None


def boardDeltas(next_visible_board, prev_full_board):
    """Return deltas of full_board relative to visible_board for visible
    board positions only.
    """

    deltas = set()
    for (row, col), nv_puyo_type in np.ndenumerate(next_visible_board):
        pf_puyo_type = prev_full_board[row, col]
        if nv_puyo_type is not pf_puyo_type:
            deltas.add(PuyoPlacement(nv_puyo_type, row, col))

    return deltas


def main():
    record = pickle.load(open("results/testing_results/0:00:09.p", "rb"))
    board_seq, nextpuyo_seq = robustClassify(record.p1clf)
    # board_seq, nextpuyo_seq = robustClassify(record.p2clf)

    board_seq = [state.board for state in board_seq]
    deducePlaySequence(board_seq, nextpuyo_seq)


if __name__ == "__main__":
    main()

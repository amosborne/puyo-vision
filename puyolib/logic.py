from collections import namedtuple
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
PlaySequence = namedtuple("PlaySequence", ["board_list", "event_list"])


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
    play_sequences = [PlaySequence(board_list=[board], event_list=[])]

    # Loop through the board sequence to determine valid play sequences.
    nextpuyo_idx = 0
    board_seq = board_seq[1:]
    for board in board_seq:
        popset = getPopSet(board)
        for play_seq in play_sequences:
            if popset:
                break
            else:
                deltas = boardDeltas(board, play_seq.board_list[-1])
                moves = possiblePuyoMoves(
                    play_seq.board_list[-1], nextpuyo_seq[nextpuyo_idx], deltas
                )
                print(moves)

    return None


def possiblePuyoMoves(board, nextpuyo, deltas):
    """Return the set of puyo moves possible given the board and deltas."""

    # Find the non-garbage deltas.
    color_deltas = [d for d in deltas if d.puyo_type is not Puyo.GARBAGE]

    # Determine the possible moves.
    # Sanity check there can be no more than 2 deltas.
    if len(color_deltas) > 2:
        raise UserWarning("More than two puyos suspected to be placed in one move.")
    elif len(color_deltas) == 2:
        if (nextpuyo[0] is None or color_deltas[0].puyo_type is nextpuyo[0]) and (
            nextpuyo[1] is None or color_deltas[1].puyo_type is nextpuyo[1]
        ):
            return PuyoMove(color_deltas[0], color_deltas[1])
        elif (
            color_deltas[0].puyo_type is nextpuyo[1]
            and color_deltas[1].puyo_type is nextpuyo[0]
        ):
            return PuyoMove(color_deltas[0], color_deltas[1])
        else:
            raise UserWarning("The two puyos placed do not match the next puyos.")
    else:
        pass

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


def getPopSet(board):
    """Return a set of puyos popping on the given board."""

    popset = set()
    for (row, col), puyo_type in np.ndenumerate(board):
        if puyo_type is Puyo.NONE:
            continue
        popgroup = set([PuyoPlacement(puyo_type, row, col)])
        while True:
            puyos_to_add = set()
            for puyo in popgroup:
                if puyo is Puyo.GARBAGE:
                    continue
                adjpuyos = getAdjPuyos(board, puyo)
                filtered_adjpuyos = [
                    p
                    for p in adjpuyos
                    if p.puyo_type is puyo.puyo_type or p.puyo_type is Puyo.GARBAGE
                ]
                puyos_to_add.update(filtered_adjpuyos)
            if popgroup >= puyos_to_add:
                break
            else:
                popgroup.update(puyos_to_add)
        if len(popgroup) >= 4:
            popset.update(popgroup)
    return popset


def getAdjPuyos(board, puyo):
    """Return the set of adjacent puyo placements to the given puyo."""

    adjset = set()
    if puyo.row > 0:
        adjset.add(PuyoPlacement(board[puyo.row - 1, puyo.col], puyo.row - 1, puyo.col))
    if puyo.row < 11:
        adjset.add(PuyoPlacement(board[puyo.row + 1, puyo.col], puyo.row + 1, puyo.col))
    if puyo.col > 0:
        adjset.add(PuyoPlacement(board[puyo.row, puyo.col - 1], puyo.row, puyo.col - 1))
    if puyo.col < 5:
        adjset.add(PuyoPlacement(board[puyo.row, puyo.col + 1], puyo.row, puyo.col + 1))
    return adjset


def main():
    record = pickle.load(open("results/testing_results/0:00:09.p", "rb"))
    board_seq, nextpuyo_seq = robustClassify(record.p1clf)
    # board_seq, nextpuyo_seq = robustClassify(record.p2clf)

    board_seq = [state.board for state in board_seq]
    deducePlaySequence(board_seq, nextpuyo_seq)


if __name__ == "__main__":
    main()

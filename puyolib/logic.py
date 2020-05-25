from collections import namedtuple, Counter
from copy import deepcopy
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
# play sequences, the choice will be random.
def deducePlaySequence(board_seq, nextpuyo_seq):
    """Given a board sequence and the corresponding next puyos, deduce the
    move sequence (including garbage and hidden row usage).
    """

    # Initialize the starting play sequence.
    board = np.empty(shape=(13, 6), dtype=Puyo)
    board.fill(Puyo.NONE)
    play_sequences = [PlaySequence(board_list=[board], event_list=[])]
    nextpuyo_idx = 0
    board_seq = board_seq[1:]
    firstPop = True

    # Loop through the board sequence to determine valid play sequences.
    for board in board_seq:
        new_play_sequences = []
        popset = getPopSet(board)

        # If the previous board was a pop...
        if not firstPop:
            for play_seq in play_sequences:
                prev_board = play_seq.board_list[-1]
                postpop_board = executePop(prev_board, popset)
                # placeholder : validate/prune play_seq, post-pop
                if not popset:
                    raise UserWarning("Last pop garbage check not implimented.")
                else:
                    new_play_sequences.append(
                        extendPlaySequence(play_seq, postpop_board)
                    )

        else:
            nextpuyos = nextpuyo_seq[nextpuyo_idx]
            for play_seq in play_sequences:
                new_play_sequences.extend(
                    newPossiblePlaySequences(play_seq, board, nextpuyos)
                )
            nextpuyo_idx += 1
            if popset:
                firstPop = False

        # Update the play sequences.
        play_sequences = new_play_sequences

    return None


def extendPlaySequence(play_seq, board, events=[]):
    """Return a new play sequence extended by the given board and events."""

    prev_board_seq = deepcopy(play_seq.board_list)
    prev_board_seq.append(board)
    prev_event_seq = deepcopy(play_seq.event_list)
    prev_event_seq.extend(events)
    return PlaySequence(board_list=prev_board_seq, event_list=prev_event_seq)


def executePop(pop_board, popset):
    """Pop the puyos in popset and return the resulting board."""

    base = np.copy(pop_board)
    for _, row, col in popset:
        base[row, col] = Puyo.NONE

    result = np.empty_like(base)
    result.fill(Puyo.NONE)
    for col_idx, puyo_col in enumerate(base.T):
        fall_idx = 0
        for row_idx, puyo in enumerate(puyo_col):
            if puyo is Puyo.NONE:
                fall_idx += 1
            else:
                result[row_idx - fall_idx, col_idx] = puyo

    return result


def newPossiblePlaySequences(play_seq, board, nextpuyos):
    """Return list of possible play sequences."""

    new_play_sequences = []
    prev_board = play_seq.board_list[-1]

    # Determine the possible moves given no pops.
    deltas = boardDeltas(board, play_seq.board_list[-1])
    moves, falls = possiblePuyoMoves(prev_board, nextpuyos, deltas)

    # Create new boards and subsequent play sequences.
    new_boards, events = createNewBoards(prev_board, moves, falls)
    for board, event in zip(new_boards, events):
        new_play_sequences.append(extendPlaySequence(play_seq, board, event))

    return new_play_sequences


def createNewBoards(prev_board, moves, falls):
    """Return a list of boards created by applying a single move."""

    if falls:
        UserWarning("New boards with garbage not implemented.")

    new_boards = []
    events = []
    for move in moves:
        base = np.copy(prev_board)
        base[move.p1.row, move.p1.col] = move.p1.puyo_type
        base[move.p2.row, move.p2.col] = move.p2.puyo_type
        new_boards.append(base)
        events.append([move])

    return new_boards, events


def possiblePuyoMoves(board, nextpuyo, deltas):
    """Return the set of puyo moves possible given the board and deltas."""

    # List of the non-garbage and garbage deltas.
    color_deltas = [d for d in deltas if d.puyo_type is not Puyo.GARBAGE]
    garbage_deltas = [d for d in deltas if d.puyo_type is Puyo.GARBAGE]

    if garbage_deltas:
        raise UserWarning("Garbage falls not implemented.")

    # Raise an error if there are more than two color deltas.
    if len(color_deltas) > 2:
        raise UserWarning("More than two puyos placed in one move.")

    # If there are exactly two color deltas, then determine the move.
    elif len(color_deltas) == 2:

        # The first move does has next puyos that are None.
        if nextpuyo[0] is None and nextpuyo[1] is None:
            return [PuyoMove(color_deltas[0], color_deltas[1])], []

        # If the next puyos match the two color deltas, return the move.
        nextpuyos = Counter(nextpuyo)
        deltacolors = Counter([cd.puyo_type for cd in color_deltas])
        if nextpuyos == deltacolors:
            return [PuyoMove(color_deltas[0], color_deltas[1])], []

        raise UserWarning("The two puyos placed do not match the next puyos.")

    # Handle single color deltas, garbage falls.
    else:
        raise UserWarning("Single color deltas not implemented.")

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
    # Load the board state sequence pre-deduction.
    record = pickle.load(open("results/testing_results/0:01:20.p", "rb"))
    board_state_seq, nextpuyo_seq = robustClassify(record.p1clf)
    # board_seq, nextpuyo_seq = robustClassify(record.p2clf)

    # Deduce the play sequence and the new board states.
    board_seq = [state.board for state in board_state_seq]
    deducePlaySequence(board_seq, nextpuyo_seq)


if __name__ == "__main__":
    main()

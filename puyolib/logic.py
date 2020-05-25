from collections import namedtuple, Counter
from copy import deepcopy
from itertools import chain, combinations
from puyolib.puyo import Puyo
import numpy as np

from puyolib.robustify import robustClassify
import pickle

"""
The board is 13 rows (12 plus the hidden row) and 6 columns, 0 indexed.
A puyo placement is a puyo type at a board position. The position may be
in row 14 (the vanish row). Only one puyo can be vanished per move.

A move is a set of puyo placements (potentially including garbage). The
resulting play sequence is a list of moves as well as the associated
list of board states. In the event that there are multiple valid play
sequences, one is chose at random.

Garbage always falls after the colored puyos are placed.
"""

PuyoPlc = namedtuple("PuyoPlacement", ["kind", "row", "col"])
PlaySeq = namedtuple("PlaySequence", ["boards", "moves"])


class PuyoLogicException(Exception):
    """Exception for logical inconsistencies during play sequence deduction."""

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def deducePlaySequence(board_seq, nextpuyo_seq):
    """Given a board sequence and the corresponding next puyos, deduce the
    move sequence (including garbage and hidden row usage).
    """

    # Initialize the starting play sequence.
    board = np.empty(shape=(13, 6), dtype=Puyo)
    board.fill(Puyo.NONE)
    play_sequences = [PlaySeq(boards=[board], moves=[])]
    nextpuyo_idx = 0
    board_seq = board_seq[1:]
    firstPop = True

    # Loop through the board sequence to determine valid play sequences.
    for board in board_seq:
        new_play_sequences = []
        popset = popSet(board)

        if not play_sequences:
            raise PuyoLogicException("No known valid play sequences.")

        # If the previous board was a pop...
        if not firstPop:
            for play_seq in play_sequences:
                prev_board = play_seq.boards[-1]
                prev_popset = popSet(prev_board)
                postpop_board = executePop(prev_board, prev_popset)
                if not popset:
                    if playSeqInvalid(board, postpop_board, islastpop=True):
                        continue
                    play_seq.boards.append(postpop_board)
                    deltas = boardDeltas(board, postpop_board)
                    certain_gbg, possible_gbg = possibleGarbage(postpop_board, deltas)
                    certain_moves = deltas | certain_gbg
                    new_play_sequences.extend(
                        extendUncertainPlaySequence(
                            play_seq, certain_moves, uncertain_moves=possible_gbg
                        )
                    )
                    for new_play_seq in new_play_sequences:
                        del new_play_seq.boards[-2]  # Hack to remove extra board.
                    firstPop = True
                else:
                    if playSeqInvalid(board, postpop_board, islastpop=False):
                        continue
                    play_seq.boards.append(postpop_board)
                    new_play_sequences.append(play_seq)

        else:
            nextpuyos = nextpuyo_seq[nextpuyo_idx]
            for play_seq in play_sequences:
                new_play_sequences.extend(
                    possiblePlaySequences(play_seq, board, nextpuyos)
                )
            nextpuyo_idx += 1
            if popset:
                firstPop = False

        # Update the play sequences.
        play_sequences = new_play_sequences
        print(len(play_sequences), play_sequences[0].moves)

    return None


def playSeqInvalid(next_board, postpop_board, islastpop):
    """Return true (invalid) if the visible area of the predicted board
    does not match the next board.
    """

    invalid = False
    try:
        deltas = boardDeltas(next_board, postpop_board)
        if not islastpop and deltas:
            invalid = True
        else:
            color_deltas = {d for d in deltas if d.kind is not Puyo.GARBAGE}
            if color_deltas:
                invalid = True
    except PuyoLogicException:
        invalid = True

    return invalid


def executePop(pop_board, popset):
    """Pop the puyos in popset and return the resulting board."""

    base = np.copy(pop_board)
    for puyo in popset:
        base[puyo.row, puyo.col] = Puyo.NONE

    result = np.empty_like(base)
    result.fill(Puyo.NONE)
    for col_idx, puyo_col in enumerate(base.T):
        fall_idx = 0
        for row_idx, kind in enumerate(puyo_col):
            if kind is Puyo.NONE:
                fall_idx += 1
            else:
                result[row_idx - fall_idx, col_idx] = kind

    return result


def possiblePlaySequences(play_seq, next_board, nextpuyos):
    """Return list of possible play sequences."""

    new_play_sequences = []
    prev_board = play_seq.boards[-1]

    # Determine the possible moves.
    deltas = boardDeltas(next_board, prev_board)
    move_set = possibleMoves(prev_board, nextpuyos, deltas)

    # Create new boards and subsequent play sequences.
    for moves in move_set:
        new_play_seq = extendPlaySequence(play_seq, moves)
        new_play_sequences.append(new_play_seq)

    return new_play_sequences


def extendUncertainPlaySequence(play_seq, certain_moves, uncertain_moves):
    """Return a list of possible play sequences where any number of uncertain
    moves may be applied in addition to the certain moves.
    """

    new_play_sequences = []
    for unc_move in powerset(uncertain_moves):
        moves = certain_moves | set(unc_move)
        new_play_sequences.append(extendPlaySequence(play_seq, moves))

    return new_play_sequences


def extendPlaySequence(play_seq, next_moves):
    """Return a new play sequence extended by the next moves."""

    boards = deepcopy(play_seq.boards)
    moves = deepcopy(play_seq.moves)
    next_board = applyMoves(boards[-1], next_moves)
    boards.append(next_board)
    moves.append(next_moves)
    return PlaySeq(boards, moves)


def applyMoves(prev_board, moves):
    """Return a new board created by applying a set of moves."""

    next_board = np.copy(prev_board)
    for move in moves:
        if move.row > 12:
            continue
        next_board[move.row, move.col] = move.kind

    return next_board


def possibleGarbage(board, gbg_deltas):
    """Return the set of possible garbage placements in the hidden row, given
    a set of garbage deltas relative to a board.
    """

    gbg_base_board = applyMoves(board, gbg_deltas)

    # Find the minimum garbage height. No more than five rows at a time.
    gbg_min_height = 5
    for col_idx, puyo_col in enumerate(gbg_base_board.T):
        gbg_col_height = 0
        for row_idx, kind in enumerate(puyo_col):
            pre_gbg = board[row_idx, col_idx]
            post_gbg = gbg_base_board[row_idx, col_idx]
            if pre_gbg is Puyo.NONE and post_gbg is Puyo.GARBAGE:
                gbg_col_height += 1
        if gbg_col_height < gbg_min_height:
            gbg_min_height = gbg_col_height

    # Find all hidden row positions that garbage could be in.
    gbg_max_height = gbg_min_height + 1
    certain_hidden_gbg = set()
    possible_hidden_gbg = set()
    for col_idx, puyo_col in enumerate(gbg_base_board.T):
        gbg_height = 0
        top_row = 0
        for row_idx, ind in enumerate(puyo_col):
            pre_gbg = board[row_idx, col_idx]
            post_gbg = gbg_base_board[row_idx, col_idx]
            if pre_gbg is Puyo.NONE and post_gbg is Puyo.GARBAGE:
                gbg_height += 1
            if post_gbg is not Puyo.NONE:
                top_row += 1
        if gbg_height > gbg_max_height:
            raise PuyoLogicException("Inconsistent garbage heights.")
        if top_row == 11 and gbg_height < gbg_max_height:
            if gbg_height < gbg_min_height:
                certain_hidden_gbg.add(PuyoPlc(kind=Puyo.GARBAGE, row=12, col=col_idx))
            else:
                possible_hidden_gbg.add(PuyoPlc(kind=Puyo.GARBAGE, row=12, col=col_idx))

    return certain_hidden_gbg, possible_hidden_gbg


def possibleColor(board, certain_move, remaining_puyos):
    """Return a list of color moves which combines the certain move (if any)
    with the possible moves for the remaining puyos.
    """

    if certain_move:
        return possibleColorConstrained(
            board, certain_move.pop(), remaining_puyos.pop()
        )

    hidden_spots, _ = invisiblePositions(board)
    move_set = set()
    for row, col in hidden_spots:
        certain_submove = PuyoPlc(kind=remaining_puyos[0], row=row, col=col)
        move_subset = possibleColorConstrained(
            board, certain_submove, remaining_puyos[1]
        )
        move_set |= move_subset

    return move_set


def possibleColorConstrained(board, certain_move, remaining_puyo):
    """Return the list of possible moves where certain move is known."""

    base_board = applyMoves(board, set([certain_move]))
    hidden_spots, vanish_spots = invisiblePositions(
        base_board, col_constraint=certain_move.col
    )

    move_set = set()
    for row, col in hidden_spots | vanish_spots:
        move = frozenset({certain_move, PuyoPlc(kind=remaining_puyo, row=row, col=col)})
        move_set.add(move)

    return move_set


def invisiblePositions(board, col_constraint=None):
    """Return the available hidden positions in the hidden and vanish rows."""

    hidden = set()
    vanish = set()
    for col_idx in range(0, 6):
        if col_constraint is not None:
            if abs(col_idx - col_constraint) > 1:
                continue

        height = columnHeight(board, col_idx)
        adjheight = adjColumnHeight(board, col_idx)

        if height == 12:
            hidden.add((height, col_idx))
            vanish.add((height + 1, col_idx))
        elif height == 13 and adjheight == 12:
            vanish.add((height, col_idx))

    return hidden, vanish


def adjColumnHeight(board, col_idx):
    """Return the taller height of the one or two adjacent columns."""

    if col_idx == 0:
        height = columnHeight(board, col_idx + 1)
    elif col_idx == 5:
        height = columnHeight(board, col_idx - 1)
    else:
        height = max(columnHeight(board, col_idx - 1), columnHeight(board, col_idx + 1))

    return height


def columnHeight(board, col_idx):
    """Return the height of the specified column."""

    height = 0
    for puyo in board.T[col_idx]:
        if puyo is not Puyo.NONE:
            height += 1

    return height


def possibleMoves(board, nextpuyo, deltas):
    """Return the set of puyo moves possible given the board and deltas."""

    color_deltas = {d for d in deltas if d.kind is not Puyo.GARBAGE}
    garbage_deltas = {d for d in deltas if d.kind is Puyo.GARBAGE}

    if len(color_deltas) > 2:
        raise PuyoLogicException("More than two puyos placed in one move.")

    # First move does not have a defined next puyo.
    nextpuyo = Counter(nextpuyo)
    if nextpuyo == Counter((None, None)):
        if len(color_deltas) < 2:
            raise PuyoLogicException("First move placed fewer than two puyos.")
        color_moves = set([frozenset(color_deltas)])

    else:
        base_move = set()
        for color_delta in color_deltas:
            if color_delta.kind not in nextpuyo:
                raise PuyoLogicException("Puyo placement does not match next puyos.")
            base_move.add(color_delta)
            nextpuyo.subtract([color_delta.kind])
            nextpuyo += Counter()  # Delete items with zero count.

        if nextpuyo:
            color_moves = possibleColor(board, base_move, set(nextpuyo.elements()))
        else:
            color_moves = set([frozenset(color_deltas)])

    if garbage_deltas:
        raise UserWarning("Garbage deltas not implimented.")
    else:
        garbage_moves = color_moves

    return garbage_moves


def boardDeltas(next_board, prev_board):
    """Return the differences between two (visible) boards."""

    deltas = set()
    for (row, col), next_kind in np.ndenumerate(next_board):
        prev_kind = prev_board[row, col]
        if next_kind is not prev_kind:
            if prev_kind is not Puyo.NONE:
                raise PuyoLogicException("An unpopped puyo has changed unexpectedly.")
            deltas.add(PuyoPlc(kind=next_kind, row=row, col=col))

    return deltas


def popSet(board):
    """Return the set of puyos popping on the given (visible) board."""

    popset = set()
    for (row, col), kind in np.ndenumerate(board):
        puyo = PuyoPlc(kind=kind, row=row, col=col)

        # Skip conditions.
        if puyo.kind is Puyo.NONE or puyo.kind is Puyo.GARBAGE:
            continue
        elif puyo in popset:
            continue
        elif puyo.row > 11:
            continue

        # Check for a new pop group.
        popgroup = set([puyo])
        while True:
            puyos_to_add = set()

            for puyo in popgroup:
                if puyo.kind is Puyo.GARBAGE:
                    continue
                adjpuyos = adjPuyos(board, puyo, type_filter=[puyo.kind, Puyo.GARBAGE])
                puyos_to_add.update(adjpuyos)

            if popgroup >= puyos_to_add:
                break
            else:
                popgroup.update(puyos_to_add)

        # At least four color puyos to a pop.
        popgroup_color_count = 0
        for puyo in popgroup:
            if puyo.kind is not Puyo.GARBAGE:
                popgroup_color_count += 1
        if popgroup_color_count >= 4:
            popset.update(popgroup)

    return popset


def adjPuyos(board, puyo, type_filter=[]):
    """Return the set of puyos adjacent to the given (visible) puyo and subject to
    the given type filter.
    """

    adjset = set()
    if puyo.row > 0:  # Below
        p = PuyoPlc(kind=board[puyo.row - 1, puyo.col], row=puyo.row - 1, col=puyo.col)
        adjset.add(p)
    if puyo.row < 11:  # Above
        p = PuyoPlc(kind=board[puyo.row + 1, puyo.col], row=puyo.row + 1, col=puyo.col)
        adjset.add(p)
    if puyo.col > 0:  # Left
        p = PuyoPlc(kind=board[puyo.row, puyo.col - 1], row=puyo.row, col=puyo.col - 1)
        adjset.add(p)
    if puyo.col < 5:  # Right
        p = PuyoPlc(kind=board[puyo.row, puyo.col + 1], row=puyo.row, col=puyo.col + 1)

    if not type_filter:
        return adjset

    filterset = set()
    for puyo in adjset:  # Filter
        for kind in type_filter:
            if puyo.kind is kind:
                filterset.add(puyo)
                break

    return filterset


def main():
    # Load the board state sequence pre-deduction.
    record = pickle.load(open("results/testing_results/0:01:20.p", "rb"))
    board_state_seq, nextpuyo_seq = robustClassify(record.p1clf)
    # board_state_seq, nextpuyo_seq = robustClassify(record.p2clf)

    # Deduce the play sequence and the new board states.
    board_seq = [state.board for state in board_state_seq]
    deducePlaySequence(board_seq, nextpuyo_seq)


if __name__ == "__main__":
    main()

from collections import Counter, namedtuple
from scipy.signal import savgol_filter, find_peaks
from pandas import Series
import numpy as np
from puyolib.vision import Puyo

"""
PUYO ROBUST CLASSIFIER
"""

# Threshold for positive identification during flicker correlation; a perfect
# flicker is approximately 0.4. The threshold is deliberately set lower so
# that corrupted flickers might still be detected. The set of all flickers
# found at a particular frame will then be self-validated given constaints on
# color and geometry inherent to the game.
_FLICKER_CORRELATION_PEAK_PROMINENCE = 0.19

# The window size used to group flickers into a common frame. The flicker
# correlation kernel is 13 frames long; the duration of the remaining pop
# animation is approximately 15 frames long. This value is selected somewhat
# arbitrarily.
_FLICKER_GROUP_FRAME_WINDOW_SIZE = 15

# Window size for selecting a puyo classification by majority at next puyo
# transitions and pop sequence flickers. Transition majority selected to
# minimize interference with garbage clouds and the next frame window (puyo
# still off-screen).
_BOARD_AT_TRANSITION_MAJORITY_WINDOW = (-3, 4)
_BOARD_AT_POPFLICKER_MAJORITY_WINDOW = (0, 8)

# Class factories for self-documentation.
Bean = namedtuple("Bean", "row col puyo_type")
Flicker = namedtuple("Flicker", "frameno bean")
PopGroup = namedtuple("PopGroup", "frameno beans")
State = namedtuple("State", "frameno board")


def correlateFlicker(raw_puyo, puyo_type):
    """Correlate the puyo classifications to the flicker animation kernel."""

    # Define the flicker kernels.
    color_kernel = [1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    grbge_kernel = [1, 1, 1, -1, 1, -1, 1, -1, 1, 0, 1, 1, 0]
    if puyo_type is Puyo.GARBAGE:
        unscaled_kernel = grbge_kernel
    elif puyo_type is not Puyo.NONE:
        unscaled_kernel = color_kernel

    # Scale the kernel to equally weight the two alternating puyo types.
    wgt_count = Counter(unscaled_kernel)
    hi_weight = 0.5 / wgt_count[1]
    lo_weight = 0.5 / wgt_count[-1]
    kernel = np.empty_like(unscaled_kernel, dtype=np.float32)
    for k, elem in enumerate(unscaled_kernel):
        if elem == 1:
            kernel[k] = hi_weight
        elif elem == -1:
            kernel[k] = -lo_weight
        else:
            kernel[k] = 0

    # Compute the correlation, squared. The rationale for squaring is
    # application specific given the alternating character of the kernel.
    signalhi = np.array([int(puyo == puyo_type) for puyo in raw_puyo])
    signallo = np.array([int(puyo == Puyo.NONE) for puyo in raw_puyo])
    signal = np.subtract(signalhi, signallo)
    jag_corr = Series(np.square(np.correlate(signal, kernel)))

    # Smooth the output.
    ave_corr = jag_corr.rolling(len(kernel), center=True).mean()
    smt_corr = savgol_filter(ave_corr, len(kernel), 3)
    return np.nan_to_num(smt_corr, 0)


def findFlickers(raw_boards):
    """Find all puyo flickers that meet a correlation threshold."""

    flickers = []
    for row, col in np.ndindex(raw_boards[0].shape):
        raw_puyo = [board[row, col] for board in raw_boards]
        for puyo_type in Puyo:
            if puyo_type is Puyo.NONE:
                continue
            flicker_corr = correlateFlicker(raw_puyo, puyo_type)
            corr_peaks, _ = find_peaks(
                flicker_corr, prominence=_FLICKER_CORRELATION_PEAK_PROMINENCE
            )
            for frameno in corr_peaks:
                flickers.append(Flicker(frameno, Bean(row, col, puyo_type)))
    return flickers


def clusterFlickers(flickers):
    """Cluster flickers into groups with the same frame number."""

    flickers.sort()
    flicker_groups = [[flickers.pop(0)]]
    for f in flickers:
        active_frame = flicker_groups[-1][-1].frameno
        if abs(f.frameno - active_frame) <= _FLICKER_GROUP_FRAME_WINDOW_SIZE:
            flicker_groups[-1].append(f)
        else:
            flicker_groups.append([f])

    pop_groups = []
    for flicker_group in flicker_groups:
        frameno = flicker_group[0].frameno
        pop_group = set()
        for flicker in flicker_group:
            pop_group.add(flicker.bean)
        pop_groups.append(PopGroup(frameno, beans=pop_group))
    return pop_groups


def validatePopGroup(board, beans):
    """TODO: Validate pop groups by color and geometry constraints."""

    final_beans = set()
    while True:
        if not beans:
            break
        start_bean = beans.pop()
        if start_bean.puyo_type is Puyo.GARBAGE:
            final_beans.add(start_bean)
            continue
        adjcolorset = crawlAdjColors(board, start_bean)
        if len(adjcolorset) >= 4:
            final_beans = final_beans | adjcolorset
        beans -= adjcolorset

    final_beans = checkMissingGarbage(board, final_beans)
    return final_beans


def crawlAdjColors(board, start_bean):
    """Return the set of adjacent beans sharing the same color."""

    adjcolorset = set([start_bean])
    while True:
        beans_to_add = set()
        for bean in adjcolorset:
            adjbeans = getAdjacentBeans(board, bean)
            for adjbean in adjbeans:
                if adjbean in adjcolorset:
                    continue
                if adjbean.puyo_type is start_bean.puyo_type:
                    beans_to_add.add(adjbean)
        if not beans_to_add:
            break
        adjcolorset = adjcolorset | beans_to_add
    return adjcolorset


def getAdjacentBeans(board, bean):
    """Return the set of adjacent beans on the board."""

    row, col = bean.row, bean.col
    adjbeanset = set()
    if row > 0:
        bean = board[row - 1, col]
        adjbeanset.add(Bean(row - 1, col, bean))
    if row < 11:
        bean = board[row + 1, col]
        adjbeanset.add(Bean(row + 1, col, bean))
    if col > 0:
        bean = board[row, col - 1]
        adjbeanset.add(Bean(row, col - 1, bean))
    if col < 5:
        bean = board[row, col + 1]
        adjbeanset.add(Bean(row, col + 1, bean))
    return adjbeanset


def checkMissingGarbage(board, beans):
    """Checks for adjacent garbage next to non-garbage popping beans."""

    updated_popgroup = beans.copy()
    for bean in beans:
        if bean.puyo_type is Puyo.GARBAGE:
            continue
        adjbeans = getAdjacentBeans(board, bean)
        for adjbean in adjbeans:
            if adjbean.puyo_type is Puyo.GARBAGE:
                updated_popgroup.add(adjbean)
    return updated_popgroup


def findPopGroups(raw_boards):
    """Find all pop groups (with their respective frames)."""

    raw_flickers = findFlickers(raw_boards)
    unvalidated_pop_groups = clusterFlickers(raw_flickers)
    return unvalidated_pop_groups


def findTransitions(raw_nextpuyo):
    """Find all frames where the next puyo pair is drawn."""

    delay = 0
    transitions = []
    for fno in range(len(raw_nextpuyo) - 2):
        if delay > 0:
            delay -= 1
            continue
        none_count = 0
        for idx in range(3):
            puyo1, puyo2 = raw_nextpuyo[fno + idx]
            if puyo1 is Puyo.NONE:
                none_count += 1
            if puyo2 is Puyo.NONE:
                none_count += 1
        if none_count >= 4:
            transitions.append(fno)
            delay = 3
    return transitions


def getPopSequence(prev_trans, next_trans, pop_groups):
    """Return the list of pop groups that occur between the two transition
       frames.
    """

    pop_sequence = []
    for pop_group in pop_groups:
        if prev_trans < pop_group.frameno < next_trans:
            pop_sequence.append(pop_group)
    return pop_sequence


def getPuyoMajority(raw_puyo):
    """Return the majority puyo classification in a sequence."""

    puyo_count = Counter(raw_puyo)
    return puyo_count.most_common(1).pop(0)


def garbageAtTransition(prev_board_state, raw_boards, trans, next_trans):
    """Find garbage majority for open positions about the transition."""

    new_board = np.copy(prev_board_state.board)
    for (row, col), puyo in np.ndenumerate(prev_board_state.board):
        if puyo is not Puyo.NONE:
            continue
        pos = round(trans + 3 * (next_trans - trans) / 4)
        raw_boards_seg = raw_boards[trans:pos]
        raw_puyo_seg = [board[row, col] for board in raw_boards_seg]
        puyo_type, count = getPuyoMajority(raw_puyo_seg)
        more_than_half = count > (len(raw_puyo_seg) / 2)
        if more_than_half and (puyo_type is Puyo.GARBAGE):
            new_board[row, col] = puyo_type
    return State(trans, new_board)


def boardAtTransition(prev_board_state, raw_boards, trans):
    """Find puyo color majority for open positions about the transition."""

    new_board = np.copy(prev_board_state.board)
    for (row, col), puyo in np.ndenumerate(prev_board_state.board):
        if puyo is not Puyo.NONE:
            continue
        neg, pos = _BOARD_AT_TRANSITION_MAJORITY_WINDOW
        raw_boards_seg = raw_boards[trans + neg : trans + pos]
        raw_puyo_seg = [board[row, col] for board in raw_boards_seg]
        puyo_type, count = getPuyoMajority(raw_puyo_seg)
        more_than_half = count > (len(raw_puyo_seg) / 2)
        if more_than_half and puyo_type is not Puyo.GARBAGE:
            new_board[row, col] = puyo_type
    return State(trans, new_board)


def executePop(pre_pop_board, beans):
    """Compute the resulting board after the pop of the given beans."""

    beans = validatePopGroup(pre_pop_board, beans)
    post_pop_board = np.empty_like(pre_pop_board)
    post_pop_board.fill(Puyo.NONE)
    for col_idx, col in enumerate(pre_pop_board.T):
        pop_idx = 0
        for row_idx, puyo in enumerate(col):
            if Bean(row_idx, col_idx, puyo) in beans:
                pop_idx += 1
            else:
                post_pop_board[row_idx - pop_idx, col_idx] = puyo
    return post_pop_board


def boardsDuringPopSequence(
    prev_board_state, raw_boards, pop_seq, next_trans, next_next_trans
):
    """Return the list of all board states during a pop sequence."""

    pop_board_states = []
    next_pop_board = np.copy(prev_board_state.board)
    for frameno, beans in pop_seq:
        # Run through the open spots to check for new beans. Important for
        # both the most recently placed beans plus any that fell out of
        # the hidden row due to the latest pop.
        for row, col, puyo_type in beans:
            if next_pop_board[row, col] is Puyo.NONE:
                next_pop_board[row, col] = puyo_type
        for (row, col), puyo in np.ndenumerate(next_pop_board):
            if puyo is not Puyo.NONE:
                continue
            neg, pos = _BOARD_AT_POPFLICKER_MAJORITY_WINDOW
            raw_boards_seg = raw_boards[frameno + neg : frameno + pos]
            raw_puyo_seg = [board[row, col] for board in raw_boards_seg]
            puyo_type, count = getPuyoMajority(raw_puyo_seg)
            more_than_half = count > (len(raw_puyo_seg) / 2)
            if more_than_half:
                next_pop_board[row, col] = puyo_type
        # Append to the list of pop board states.
        pop_board_states.append(State(frameno, next_pop_board))
        # Execute the pop to derive the next pop board.
        next_pop_board = executePop(next_pop_board, beans)
    # Add the final board at the end of the pop sequence, inclusive of
    # any garbage that may have fallen at the end of the sequence.
    final_board_state = State(next_trans, next_pop_board)
    if next_next_trans:
        final_board_state = garbageAtTransition(
            final_board_state, raw_boards, next_trans, next_next_trans
        )
    pop_board_states.append(final_board_state)
    return pop_board_states


def buildBoardSequence(raw_boards, transitions, pop_groups):
    """Build the sequence of boards (with their respective frames)."""

    # Initialize the first board in the list of board states to be empty.
    initial_board = np.empty_like(raw_boards[0], dtype=Puyo)
    initial_board.fill(Puyo.NONE)
    board_states = [State(0, initial_board)]
    for idx, this_trans in enumerate(transitions):
        prev_board_state = board_states[-1]
        # Get the frames of the previous and next transitions.
        if idx == 0:
            prev_trans = 0
        else:
            prev_trans = transitions[idx - 1]
        if idx == (len(transitions) - 1):
            next_trans = len(raw_boards) - 1
            next_next_trans = None
        elif idx == (len(transitions) - 2):
            next_trans = transitions[idx + 1]
            next_next_trans = len(raw_boards) - 1
        else:
            next_trans = transitions[idx + 1]
            next_next_trans = transitions[idx + 2]
        # Get whether the previous and next frame windows had pops and process.
        prev_pop_sequence = getPopSequence(prev_trans, this_trans, pop_groups)
        next_pop_sequence = getPopSequence(this_trans, next_trans, pop_groups)
        if not next_pop_sequence:
            prev_board_state = garbageAtTransition(
                prev_board_state, raw_boards, this_trans, next_trans
            )
        if not prev_pop_sequence:
            prev_board_state = boardAtTransition(
                prev_board_state, raw_boards, this_trans
            )
            board_states.append(prev_board_state)
        if next_pop_sequence:
            pop_board_states = boardsDuringPopSequence(
                prev_board_state,
                raw_boards,
                next_pop_sequence,
                next_trans,
                next_next_trans,
            )
            board_states += pop_board_states
    return board_states


def robustClassify(raw_clf):
    """Return the board state sequence. Raw classifications are assumed to
       begin after the first puyo pair is drawn and ends at an arbitrary time
       (but not long after game loss). Final puyo pair placement during a
       fatal blow will not be captured.

       Raw classifications are received as a list of tuples:
          * Index is the frame number.
          * First element is a numpy array representing the board.
          * Second element is a tuple of the next puyo pair.

       Board sequence is returned as a list of named tuples:
          * board_seq[k].frameno
          * board_seq[k].board
    """

    raw_boards, raw_nextpuyo = tuple(zip(*raw_clf))
    pop_groups = findPopGroups(raw_boards)
    transitions = findTransitions(raw_nextpuyo)
    board_seq = buildBoardSequence(raw_boards, transitions, pop_groups)
    return board_seq

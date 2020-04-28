from collections import Counter, defaultdict, namedtuple
from scipy.signal import savgol_filter, find_peaks
from pandas import Series
import numpy as np
from puyocv import Puyo

from matplotlib import pyplot as plt

"""
PUYO ROBUST CLASSIFIER

The puyo computer vision module will return a raw classification for each frame of video.
This module analyzes these raw classifications to robustly produce a sequence of board states
which correspond to new puyo placements or puyo pop sequences.

The most reliable elements detected by the raw classifications are:
   1) The ~4 frame 'transition' when the next puyo is drawn.
   2) The ~10 frame 'flickering' of puyos when undergoing a pop. In fact, pops are a good
      time to evaluate the entire board state because no puyos are falling or otherwise animating.

Of course even those elements aren't always perfectly detected; animations which overlay or
otherwise distort the puyos will often result in mis-classifications. Characteristics of the
game may be further leveraged to be robust to these mis-classifications:
   1) When the next puyo is drawn, both puyos in the pair should register a change to empty.
   2) Puyos must pop in groups of like color of 4 or more (and garbage must be adjacent).
   3) In general, the majority classification over many frames is the correct classification.

When all pops and next puyo transitions are identified, the entire board state sequence can
be derived by the following steps:
   1) At any transition if the frames between the prior transition and the next transition
      contain no pops, determine the majority puyo class in each empty board position for
      each frame window on either side of the transition and look for changes from empty.
   2) At any transition if the prior frame window has no pops but the later frame window
      does, determine the majority puyo class in each empty board position for the frames
      during the pop flickering. Disambiguate the placement of two puyo pairs by which
      board position held the final puyo classification for more frames since the previous
      transition.
   3) At any transition if both the prior frame window and later frame window has pops,
      compare the board state between the final pop of the earlier sequence and the first
      pop of the later sequence (again by using majority classification on empty frames).

Placement frames and pop frames are compiled and returned as the robust classification. It
is assumed that the first frame is the start of the game and the final frame is prior to any 
end game animation.
"""

# Threshold for positive identification during flicker correlation; a perfect flicker is
# approximately 0.4. The threshold is deliberately set lower so that corrupted flickers
# might still be detected. The set of all flickers found at a particular frame will then
# be self-validated given constaints on color and geometry inherent to the game.
_FLICKER_CORRELATION_PEAK_PROMINENCE = 0.2

# The window size used to group flickers into a common frame. The flicker correlation
# kernel is 13 frames long; the duration of the remaining pop animation is approximately
# 15 frames long. This value is selected somewhat arbitrarily.
_FLICKER_GROUP_FRAME_WINDOW_SIZE = 15

# Class factories for self-documentation.
Bean     = namedtuple('Bean', 'row col puyo_type')
Flicker  = namedtuple('Flicker', 'frameno bean')
PopGroup = namedtuple('PopGroup', 'frameno beans')
State    = namedtuple('State', 'frameno board')

def correlateFlicker(raw_puyo, puyo_type):
    """Correlate the puyo classifications to the flicker animation kernel."""
    
    # Define the flicker kernels.
    color_kernel = [ 1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1 ]
    grbge_kernel = [ 1, 1, 1,-1, 1,-1, 1,-1, 1, 0, 1, 1, 0 ]
    if   puyo_type is     Puyo.GARBAGE:
        unscaled_kernel = grbge_kernel
    elif puyo_type is not Puyo.NONE:
        unscaled_kernel = color_kernel

    # Scale the kernel to equally weight the two alternating puyo types.
    wgt_count = Counter(unscaled_kernel)
    hi_weight = 0.5/wgt_count[ 1]
    lo_weight = 0.5/wgt_count[-1]
    kernel    = np.empty_like(unscaled_kernel, dtype=np.float32)
    for k,elem in enumerate(unscaled_kernel):
        if   elem ==  1: kernel[k] =  hi_weight
        elif elem == -1: kernel[k] = -lo_weight
        else:            kernel[k] =  0

    # Compute the correlation, squared. The rationale for squaring is
    # application specific given the alternating character of the kernel.
    signalhi = np.array([int(puyo==puyo_type) for puyo in raw_puyo])
    signallo = np.array([int(puyo==Puyo.NONE) for puyo in raw_puyo])
    signal   = np.subtract(signalhi, signallo)
    jag_corr = Series(np.square(np.correlate(signal, kernel)))

    # Smooth the output.
    ave_corr = jag_corr.rolling(len(kernel),center=True).mean()
    smt_corr = savgol_filter(ave_corr, len(kernel), 3)
    return     np.nan_to_num(smt_corr, 0)

def findFlickers(raw_boards):
    """Find all puyo flickers that meet a correlation threshold."""

    flickers = []
    for row,col in np.ndindex(raw_boards[0].shape):
        raw_puyo = [board[row,col] for board in raw_boards]
        for puyo_type in Puyo:
            if puyo_type is Puyo.NONE:
                continue
            flicker_corr = correlateFlicker(raw_puyo, puyo_type)
            corr_peaks,_ = find_peaks(flicker_corr, prominence=_FLICKER_CORRELATION_PEAK_PROMINENCE)
            for frameno in corr_peaks:
                flickers.append(Flicker(frameno, Bean(row,col,puyo_type)))
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
        pop_groups.append(PopGroup(frameno, pop_group))
    return pop_groups

def validatePopGroups(pop_groups):
    """TODO: Validate pop groups by color and geometry constraints."""
    
    return pop_groups

def findPopGroups(raw_boards):
    """Find all pop groups (with their respective frames)."""
    
    raw_flickers = findFlickers(raw_boards)
    unvalidated_pop_groups = clusterFlickers(raw_flickers)
    validated_pop_groups = validatePopGroups(unvalidated_pop_groups)
    return validated_pop_groups

def findTransitions(raw_nextpuyo):
    """Find all frames where the next puyo pair is drawn."""
    
    both_blank = []
    for puyo1,puyo2 in raw_nextpuyo:
        if puyo1 is Puyo.NONE and puyo2 is Puyo.NONE:
            both_blank.append(True)
        else:
            both_blank.append(False)
    transitions = []
    isblank = False
    for idx,b in enumerate(both_blank):
        if b and not isblank:
            transitions.append(idx)
            isblank = True
        elif not b and isblank:
            isblank = False
    return transitions

def getPopSequence(prev_trans,next_trans,pop_groups):
    """Return the list of pop groups that occur between the two transition frames."""
    
    pop_sequence = []
    for pop_group in pop_groups:
        if prev_trans < pop_group.frameno < next_trans:
            pop_sequence.append(pop_group)
    return pop_sequence

def getPuyoMajority(raw_puyo):
    puyo_count = Counter(raw_puyo)
    return puyo_count.most_common(1).pop(0)

_BOARD_AT_TRANSITION_MAJORITY_WINDOW = (-3,4)

def garbageAtTransition(prev_board_state, raw_boards, trans, next_trans):
    """Find garbage majority for open positions about the transition."""

    new_board = np.copy(prev_board_state.board)
    for (row,col),puyo in np.ndenumerate(prev_board_state.board):
        if puyo is not Puyo.NONE:
            continue
        pos = round((next_trans + trans)/2)
        raw_boards_segment = raw_boards[trans:trans+pos]
        raw_puyo_segment = [board[row,col] for board in raw_boards_segment]
        puyo_type, count = getPuyoMajority(raw_puyo_segment)
        if (count > (len(raw_puyo_segment)/2)) and (puyo_type is Puyo.GARBAGE):
            new_board[row,col] = puyo_type
    return State(trans, new_board)

def boardAtTransition(prev_board_state, raw_boards, trans):
    """Find puyo color majority for open positions about the transition."""

    new_board = np.copy(prev_board_state.board)
    for (row,col),puyo in np.ndenumerate(prev_board_state.board):
        if puyo is not Puyo.NONE:
            continue
        neg, pos = _BOARD_AT_TRANSITION_MAJORITY_WINDOW
        raw_boards_segment = raw_boards[trans+neg:trans+pos]
        raw_puyo_segment = [board[row,col] for board in raw_boards_segment]
        puyo_type, count = getPuyoMajority(raw_puyo_segment)
        if count > (len(raw_puyo_segment)/2):
            new_board[row,col] = puyo_type
    return State(trans, new_board)

def buildBoardSequence(raw_boards, transitions, pop_groups):
    """Build the sequence of boards (with their respective frames)."""

    # Initial the first board in the list of board states to be empty.
    initial_board = np.empty_like(raw_boards[0], dtype=Puyo)
    initial_board.fill(Puyo.NONE)
    board_states = [State(0, initial_board)]
    for idx,this_trans in enumerate(transitions):
        prev_board_state = board_states[-1]
        # Get the indices of the previous and next transitions.
        if idx == 0:
            prev_trans = 0
            next_trans = transitions[idx+1]
        elif idx == (len(transitions) - 1):
            prev_trans = transitions[idx-1]
            next_trans = len(raw_boards) - 1
        else:
            prev_trans = transitions[idx-1]
            next_trans = transitions[idx+1]
        # Get whether the previous and next frame windows had pops and process.
        prev_pop_sequence = getPopSequence(prev_trans, this_trans, pop_groups)
        next_pop_sequence = getPopSequence(this_trans, next_trans, pop_groups)
        if not next_pop_sequence:
            prev_board_state = garbageAtTransition(prev_board_state, raw_boards, this_trans, next_trans)
        if not prev_pop_sequence:
            this_board_state = boardAtTransition(prev_board_state, raw_boards, this_trans)
            board_states.append(this_board_state)
        if next_pop_sequence:
            pass # generate pop boards by assessing majorities at +7 window about pop frame
    return board_states

import puyodebug
import cv2

def robustClassify(raw_clf):
    """Return the board state sequence. Raw classifications are assumed to begin after
       the first puyo pair is drawn and ends at an arbitrary time (but not long after
       game loss). Final puyo pair placement during a fatal blow will not be captured.
    
       Raw classifications are received as a list of tuples:
          * Index is the frame number.
          * First element is a numpy array representing the board. (board[row][col] = puyo)
          * Second element is a tuple of the next puyo pair.
    """
    
    raw_boards, raw_nextpuyo = tuple(zip(*raw_clf))
    pop_groups = findPopGroups(raw_boards)
    transitions = findTransitions(raw_nextpuyo)
    board_seq = buildBoardSequence(raw_boards,transitions,pop_groups)
    for _,board in board_seq:
        img = puyodebug.plotBoardState(board)
        cv2.imshow('',img)
        cv2.waitKey(0)
    return None

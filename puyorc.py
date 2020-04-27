from collections import defaultdict, Counter
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

_FLICKER_CORRELATION_PEAK_PROMINENCE = 0.2 # NOTE: A perfect flicker is approx. 0.4
_FLICKER_GROUP_MAX_GAP_SIZE = 15 # NOTE: Flicker kernel is of length 13

def correlateFlicker(raw_puyo, puyo_type):
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
    # Compute the correlation, squared.
    signalhi = np.array([int(x==puyo_type) for x in raw_puyo])
    signallo = np.array([int(x==Puyo.NONE) for x in raw_puyo])
    signal   = np.subtract(signalhi, signallo)
    jag_corr = Series(np.square(np.correlate(signal, kernel))) # this is a bit of a hack.
    # Smooth the output.
    ave_corr = jag_corr.rolling(len(kernel)).mean()
    smt_corr = savgol_filter(ave_corr, len(kernel), 3)
    return     np.nan_to_num(smt_corr, 0)

def findFlickers(raw_boards):
    flickers = list()
    for row,col in np.ndindex(raw_boards.shape[1:]):
        raw_puyo = raw_boards[:,row,col]
        for puyo_type in Puyo:
            if puyo_type is Puyo.NONE: continue
            flicker = correlateFlicker(raw_puyo, puyo_type)
            peaks, _ = find_peaks(flicker, prominence=_FLICKER_CORRELATION_PEAK_PROMINENCE)
            for pk in peaks:
                flickers.append((pk,row,col,puyo_type))
    return flickers

def clusterFlickers(flickers):
    flickers.sort(key=lambda f: f[0])
    flicker_groups = [[flickers[0]]]
    for f in flickers[1:]:
        if abs(f[0] - flicker_groups[-1][-1][0]) <= _FLICKER_GROUP_MAX_GAP_SIZE:
            flicker_groups[-1].append(f)
        else:
            flicker_groups.append([f])
    aligned_flickers = defaultdict(list)
    for group in flicker_groups:
        frameno = group[0][0]
        for puyo in group:
            aligned_flickers[frameno].append(puyo[1:])
    return aligned_flickers

def validateFlickers(board_flickers):
    return board_flickers # TODO: Check for false positives given loose thresholds.

def boardFlickerList(raw_boards):
    raw_boards = np.asarray(raw_boards)
    raw_flickers = findFlickers(raw_boards)
    board_flickers = clusterFlickers(raw_flickers)
    board_flickers = validateFlickers(board_flickers)
    return board_flickers # dict[frameno]:[(row,col,puyo)]

def transitionList(raw_nextpuyo):
    both_blank = []
    for (b1,b2) in raw_nextpuyo:
        if b1 is Puyo.NONE and b2 is Puyo.NONE:
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

def hasPopSequence(early_frame,later_frame,board_flickers):
    popsequence = {}
    for popframe,poplist in board_flickers.items():
        if early_frame < popframe < later_frame:
            popsequence[popframe] = poplist
    if popsequence: return True, popsequence
    else:           return False, None

_PREVIOUS_PLACEMENT_NEXT_WINDOW_SHARE_THRESHOLD = 0.9 # not clear how well tuned this is.
    
def noPopBoardDetermination(prev_board_state, raw_boards, transitions):
    next_board_state = np.copy(prev_board_state)
    lastt,t,nextt = transitions
    for (row,col), puyo in np.ndenumerate(prev_board_state):
        if puyo is not Puyo.NONE: continue
        raw_puyo = raw_boards[:,row,col]
        prev_color = Counter(raw_puyo[lastt:t])
        prev_color = prev_color.most_common(1)[0][0]
        next_color = Counter(raw_puyo[t:nextt])
        next_color = next_color.most_common(1)[0]
        if prev_color is not Puyo.NONE:
            next_board_state[row,col] = prev_color
        elif next_color[0] is not Puyo.NONE:
            share = next_color[1]/(nextt-t)
            if share > _PREVIOUS_PLACEMENT_NEXT_WINDOW_SHARE_THRESHOLD:
                next_board_state[row,col] = next_color[0]
    return next_board_state

def buildBoardSequence(raw_boards, transitions, board_flickers):
    raw_boards = np.asarray(raw_boards)
    for idx,t in enumerate(transitions[:-1]):
        if idx == 0:
            board_state = [(t, np.empty_like(raw_boards[0], dtype=Puyo))]
            board_state[0][1].fill(Puyo.NONE)
        else:
            lastt = transitions[idx-1]
            nextt = transitions[idx+1]
            hasprepop,  prepoplist  = hasPopSequence(lastt,t,board_flickers)
            haspostpop, postpoplist = hasPopSequence(t,nextt,board_flickers)
            if hasprepop and haspostpop: break # UPDATE
            elif hasprepop:  break # UPDATE
            elif haspostpop: break # UPDATE
            else:
                board_state.append( (t, noPopBoardDetermination(board_state[-1][1],
                                                                raw_boards,
                                                                (lastt,t,nextt))))
    return board_state

import puyodebug
import cv2

# Raw classifications are received as a list (raw_clf) of tuples, index is the frame number.
#   (1) First element is a numpy array reprenting the board. board[row][col] = Puyo
#   (2) Second element is a tuple of the next Puyo pair.
def robustClassify(raw_clf):
    raw_boards, raw_nextpuyo = tuple(zip(*raw_clf))
    board_flickers = boardFlickerList(raw_boards)
    transitions = transitionList(raw_nextpuyo)
    board_seq = buildBoardSequence(raw_boards,transitions,board_flickers)
    for _,board in board_seq:
        img = puyodebug.plotBoardState(board)
        cv2.imshow('',img)
        cv2.waitKey(0)
    return None

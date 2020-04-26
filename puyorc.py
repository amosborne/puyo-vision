from collections import defaultdict, Counter
from warnings import warn
from math import ceil, floor
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

# Pop animation flicker detection constants.
_COLOR_FLICKER_FRAME_COUNT = 9
_GRBGE_FLICKER_FRAME_COUNT = 7
_COLOR_FLICKER_ERROR_ALLWD = (3,2)
_GRBGE_FLICKER_ERROR_ALLWD = (2,1)

def pruneBoardFlickers(board_flickers):
    return board_flickers # dict: [frameno]=[(row,col,Puyo)]

def alignBoardFlickers(board_flickers):
    return board_flickers # dict: [frameno]=[(row,col,Puyo)]

def evalFlicker(seq,err):
    fc = len(seq)
    hi = Counter(seq[0:fc:2])
    lo = Counter(seq[1:(fc-1):2])
    clr = hi.most_common(1)[0][0]
    err_hi = ceil(fc/2) - hi[clr]
    err_lo = floor(fc/2) - lo[Puyo.NONE]
    if (err_hi < err[0]) and (err_lo < err[1]):
        return (clr,err_hi+err_lo)
    return (None,0)

# How many times a puyo flickers during a pop depends on the puyo type (garbage or not), and also
# varies a little depending the animation. This function will scan the raw classification for a
# a single puyo to meet either of the following constraints:
#   (1) For 7 consecutive frames, the puyo alternates between garbage and empty (3 errors allowed)
#   (2) For 9 consecutive frames, the puyo alternates between colored and empty (5 errors allowed)
# The error thresholds are selected to be forgiving. The puyo color is taken to be the majority
# color among any color classification errors within the sequence. This function will flag any
# flickering puyo with errors to be further validated by adjacent flickering puyos later. The same
# flicker sequence may also be flagged at different frames, this will also be cleaned later.
def puyoFlickerList(raw_puyo):
    puyo_flickers = []
    total_frames = len(raw_puyo)
    for frameno, _ in enumerate(raw_puyo[:-_COLOR_FLICKER_FRAME_COUNT]): # bounds not critical
        # Evaluate for garbage flicker.
        seq = raw_puyo[frameno:(frameno+_GRBGE_FLICKER_FRAME_COUNT):2]
        puyo, error = evalFlicker(seq,_GRBGE_FLICKER_ERROR_ALLWD)
        if puyo is Puyo.GARBAGE:
            puyo_flickers.append((frameno,puyo,error))
            continue
        # Evaluate for color flicker if it wasn't garbage flicker.
        seq = raw_puyo[frameno:(frameno+_COLOR_FLICKER_FRAME_COUNT):2]
        puyo, error = evalFlicker(seq,_COLOR_FLICKER_ERROR_ALLWD)
        if puyo is not None and puyo is not Puyo.NONE:
            puyo_flickers.append((frameno,puyo,error))
    return puyo_flickers # list: (frameno, Puyo, errors)

def boardFlickerList(raw_boards):
    raw_boards = np.asarray(raw_boards)
    board_flickers = defaultdict(list)
    for row,col in np.ndindex(raw_boards.shape[1:]):
        puyo_flickers = puyoFlickerList(raw_boards[:,row,col])
        for (frameno, puyo, errors) in puyo_flickers:
            board_flickers[frameno].append((row,col,puyo,errors))        
    board_flickers = alignBoardFlickers(board_flickers)
    board_flickers = pruneBoardFlickers(board_flickers)
    return board_flickers # dict: [frameno]=[(row,col,Puyo,errors)]

# Raw classifications are received as a list (raw_clf) of tuples, index is the frame number.
#   (1) First element is a numpy array reprenting the board. board[row][col] = Puyo
#   (2) Second element is a tuple of the next Puyo pair.
def robustClassify(raw_clf):
    raw_boards, raw_nextpuyo = tuple(zip(*raw_clf))
    board_flickers = boardFlickerList(raw_boards)

    plt.subplots()
    for frameno,flickers in board_flickers.items():
        plt.plot(frameno,len(flickers),'x')
    plt.show()
    
    return board_flickers # list: (frameno, board, nextpuyo, ispop)

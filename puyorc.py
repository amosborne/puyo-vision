from collections import defaultdict, Counter
from warnings import warn
from math import ceil, floor
import numpy as np
from puyocv import Puyo
from matplotlib import pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from copy import deepcopy

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
_COLOR_FLICKER_FRAME_COUNT = 5
_GRBGE_FLICKER_FRAME_COUNT = 5
_COLOR_FLICKER_ERROR_ALLWD = (1,1)
_GRBGE_FLICKER_ERROR_ALLWD = (1,1)
_MAX_FLICKER_WINDOW = 20

def pruneBoardFlickers(board_flickers):
    return board_flickers # dict: [frameno]=[(row,col,Puyo,errors)]

# Flickers of a puyo with the same class and location within the same frame window are
# greedily condensed to the earliest frame.
def alignPuyoFlickers(board_flickers):
    aligned_puyo_flickers = defaultdict(list)
    # For each frame marked for flickering...
    for frameno,flicker_list in board_flickers.items():
        # Loop through each puyo that is flickering...
        for puyo_flicker in flicker_list:
            iscovered = False
            # Loop through the aligned flickering dictionary...
            for al_frameno,al_flicker_list in aligned_puyo_flickers.items():
                # Loop through each puyo that is flickering aligned...
                for al_puyo_flicker in al_flicker_list:
                    within_window = frameno < (al_frameno + _MAX_FLICKER_WINDOW)
                    same_puyo = puyo_flicker[0:3] == al_puyo_flicker[0:3]
                    # If it's a different puyo or it's outside the window, add it.
                    if within_window and same_puyo:
                        iscovered = True
            if not iscovered:
                aligned_puyo_flickers[frameno].append(puyo_flicker)
    return aligned_puyo_flickers # dict: [frameno]=[(row,col,Puyo,errors)]

def alignGroupFlickers(board_flickers):
    
    return board_flickers

def alignBoardFlickers(board_flickers):
    aligned_puyo_flickers = alignPuyoFlickers(board_flickers)
    aligned_group_flickers = alignGroupFlickers(aligned_puyo_flickers)
    return aligned_group_flickers

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

# Lets try an alternate solution:
# 1) for each position and each puyo type (non-empty), apply the scrolling flicker to the entire
#    frame sequence and determine the correlation. threshold to some value, normalize to 1.
# 2) multiply the normalized correlation by another (unnormalized) for the same color or garbage.
#    threshold again; if exceeded then they are of the same pop. continue recursively.

_FLICKER_THRESHOLD = 0.3

def correlateFlicker(raw_puyo, puyo_type):
    # Define the flicker kernels.
    color_kernel = [ 1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1, 0 ]
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
    jag_corr = pd.Series(np.square(np.correlate(signal, kernel)))
    # Smooth the output.
    ave_corr = jag_corr.rolling(len(kernel)).mean()
    smt_corr = savgol_filter(ave_corr, len(kernel), 3)
    return     np.nan_to_num(smt_corr, 0)

def evalFlickerThreshold(raw_puyo, puyo_type, matching_flicker=None):
    flicker = correlateFlicker(raw_puyo, puyo_type)
    if matching_flicker is None:
        matching_flicker = np.ones_like(flicker, dtype=np.float32)
    match = np.multiply(flicker, matching_flicker)
    maxf = np.max(match)
    if maxf > _FLICKER_THRESHOLD:
        match_norm = np.divide(match, maxf)
       # plt.subplots()
      #  plt.plot(match)
     #   plt.show()
        return True, match_norm
    #print('test')
    return False, []

def getAllPops(raw_boards):
    puyos_pos = set()
    for row,col in np.ndindex(raw_boards.shape[1:]):
        raw_puyo = raw_boards[:,row,col]
        for puyo_type in Puyo:
            if puyo_type is Puyo.NONE: continue
            if evalFlickerThreshold(raw_puyo, puyo_type)[0]:
                puyos_pos.add((row,col,puyo_type))
    return frozenset(puyos_pos)

def findFlickerGroups(raw_boards, remaining_puyo_pops=None, working_flicker=None):
    if remaining_puyo_pops is None:
        remaining_puyo_pops = getAllPops(raw_boards)
    print(len(remaining_puyo_pops))
    # Recursive loop.
    flicker_groups = set()
    for puyo_pop in remaining_puyo_pops:
        row,col,clr = puyo_pop
        raw_puyo = raw_boards[:,row,col]
        popmatch, flicker = evalFlickerThreshold(raw_puyo, clr, working_flicker)
        
            ## DEBUG
            #print((row,col,puyo_type))
            #plt.subplots()
            #plt.plot(flicker)
            #plt.show()
            ##
            
        if popmatch:
            next_remaining_puyo_pops = remaining_puyo_pops - frozenset([puyo_pop])
            print(puyo_pop,next_remaining_puyo_pops)
            recurse_results =  findFlickerGroups(raw_boards,
                                                 remaining_puyo_pops=next_remaining_puyo_pops,
                                                 working_flicker=flicker)
            #print(recurse_results)
            for result in recurse_results:
                flicker_groups.add(frozenset([member]) | result)
                    
    #print(flicker_groups)                
    return frozenset(flicker_groups)
    
def boardFlickerList(raw_boards):
    raw_boards = np.asarray(raw_boards)
    # Construct a frozen set of all puyo board positions.
    flicker_groups = findFlickerGroups(raw_boards)
        
    return None

# Raw classifications are received as a list (raw_clf) of tuples, index is the frame number.
#   (1) First element is a numpy array reprenting the board. board[row][col] = Puyo
#   (2) Second element is a tuple of the next Puyo pair.
def robustClassify(raw_clf):
    raw_boards, raw_nextpuyo = tuple(zip(*raw_clf))
    board_flickers = boardFlickerList(raw_boards)

    #print(board_flickers[110])
    #print(board_flickers[114])
    #print(board_flickers[117])
    # print(board_flickers[115])
    # print(board_flickers[117])
    # print(board_flickers[124])
    
    #plt.subplots()
    #for frameno,flickers in board_flickers.items():
    #    plt.plot(frameno,len(flickers),'x')
    #plt.show()
    
    return board_flickers # list: (frameno, board, nextpuyo, ispop)

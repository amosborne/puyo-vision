from collections import defaultdict
import numpy as np
from puyocv import Puyo

"""
PUYO ROBUST CLASSIFIER

The puyo computer vision module will return a raw classification for each frame of video.
This module analyzes these raw classifications to robustly produce a sequence of board states
which correspond to new puyo placements or puyo pop sequences.

The most reliable elements detected by the raw classifications are:
   1) The 4 frame 'transition' when the next puyo is drawn.
   2) The 10 frame 'flickering' of puyos when undergoing a pop.

Of course even those elements aren't always perfectly detected; animations which overlay or
otherwise distort the puyos will often result in mis-classifications. Characteristics of the
game may be further leveraged to be robust to these mis-classifications:
   1) When the next puyo is drawn, both puyos in the pair should register a change to empty.
   2) Puyos must pop in groups of like color of 4 or more (and garbage must be adjacent).

When all pops are identified, the complete board state can be resolved. Take the first row
of puyos, for example:
   1) If no flickering is seen, then there was no pop. For all frames, compute the two most
      common classifications. For those two classifications, locate the frame where if each
      side of the frame (in time) is labelled the maximum number of classifications are
      correct. Correlate this placement frame location with the next puyo transition frame.
   2) If there is flickering, then there was a pop. Break the frame into segments and for 
      each segment calculate the placement frame as above.

Rows above the first row are slight more complicated; to determine the placement frame, the
pops in rows below (but in the same column) must also be included in the segmentation.

Placement frames and pop frames are compiled and returned as the robust classification. It
is assumed that the first frame is the start of the game and the final frame is prior to any 
end game animation.
"""

def pruneBoardFlickers(board_flickers):
    return board_flickers # dict: [frameno]=[(row,col,Puyo)]

def alignBoardFlickers(board_flickers):
    return board_flickers # dict: [frameno]=[(row,col,Puyo)]

def puyoFlickerList(raw_puyo):
    return [] # list: (frameno, Puyo)

def boardFlickerList(raw_boards):
    raw_boards = np.asarray(raw_boards)
    board_flickers = defaultdict(list)
    for row,col in np.ndindex(raw_boards.shape):
        puyo_flickers = puyoFlickerList(raw_boards[:,row,col])
        for frameno, puyo in puyo_flickers:
            board_flickers[frameno].append((row,col,puyo))        
    board_flickers = alignBoardFlickers(board_flickers)
    board_flickers = pruneBoardFlickers(board_flickers)
    return board_flickers # dict: [frameno]=[(row,col,Puyo)]

# Raw classifications are received as a list (raw_clf) of tuples, index is the frame number.
#   (1) First element is a numpy array reprenting the board. board[row][col] = Puyo
#   (2) Second element is a tuple of the next Puyo pair.
def robustClassify(raw_clf):
    raw_boards, raw_nextpuyo = tuple(zip(*raw_clf))
    board_flickers = boardFlickerList(raw_boards)
    return None # list: (frameno, board, nextpuyo, ispop)

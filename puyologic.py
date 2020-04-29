import pickle
import copy
import numpy as np
from puyocv import Puyo
from matplotlib import pyplot as plt
import puyodebug

import cv2

def plotClf(clf,row=1,stable_clf=None):
    fig, axes = plt.subplots(6,sharex=True)
    fig.suptitle("ROW " + str(row))
    nextpuyos = [x[1] for x in clf]
    transitions = getTransitionFrames(clf)
    puyos = [x[0][row-1] for x in clf]
    for idx, ax in enumerate(axes):
        p = np.array([x[idx] for x in puyos])
        ax.plot(p)
        ax.plot(transitions,p[transitions],'rx')
    if stable_clf is not None:
        puyos = [x[0][row-1] for x in stable_clf]
        for idx, ax in enumerate(axes):
            p = [x[idx] for x in puyos]
            ax.plot(p)

def flashStabilize(clf,stabilization):
    stable_clf = copy.deepcopy(clf)
    boards = [x[0] for x in stable_clf]
    for b_idx, b in enumerate(boards[:-2]):
        for r_idx, r_puyos in enumerate(b):
            for c_idx, puyo in enumerate(r_puyos):
                nextpuyos = [x[r_idx][c_idx] for x in boards[b_idx+1:b_idx+stabilization]]
                if len(set(nextpuyos)) > 1:
                    boards[b_idx+1][r_idx][c_idx] = puyo
    return stable_clf

def getTransitionFrames(clf):
    nextpuyos = [x[1] for x in clf]
    transitions = []
    for idx, (a,b) in enumerate(nextpuyos[1:-1]):
        pasta = nextpuyos[idx-1][0]
        nexta = nextpuyos[idx+1][0]
        pastb = nextpuyos[idx-1][1]
        nextb = nextpuyos[idx+1][1]
        agood = ((pasta == a) or (nexta == a)) and (a == Puyo.NONE)
        bgood = ((pastb == b) or (nextb == b)) and (b == Puyo.NONE)
        if transitions:
            igood = idx > (transitions[-1] + 10) # magic number
        else:
            igood = True
        if agood and bgood and igood:
            transitions.append(idx)
    return transitions

def getTransitionBoards(clf,transitions):
    boards = np.array([x[0] for x in clf])
    transition_boards = np.stack(boards[transitions])
    return transition_boards

def boardPopped(thisboard,lastboard):
    # The board "popped" this board is not a strict addition (or atleast the same) as the last board.
    # The current function code does not account for the hidden row...
    popped = False
    for c_idx, col in enumerate(lastboard.T):
        for r_idx, puyo in enumerate(col):
            wasempty = (puyo == Puyo.NONE)
            isempty = (thisboard[r_idx,c_idx] == Puyo.NONE)
            if isempty and not wasempty:
                popped = True
                break
    return popped

def findLastFlicker(clf,idxrange):
    # A group of puyos that is popping will flicker between their class and empty ~5 times prior to
    # dissapearing. This function will search the non-flash stabilized classifications for the most
    # recently flashing puyos and output the frame when the flashing began. This function should
    # eventually incorporate the dissapearance of garbage puyos on the first frame...
    flicker_length = 7
    frameno = idxrange[0]
    garbage = None
    flicker = None
    boards = np.array([x[0] for x in clf])
    for row,col in np.ndindex(boards[0].shape):
        # For each puyo position on the board...
        for idx in range(idxrange[0],idxrange[1]-flicker_length):
            # Sweep forward in time...
            flicker_clr = boards[idx][row,col]
            lastclass = flicker_clr
            isflicker = flicker_clr is not Puyo.NONE
            for i_idx in range(idx+1,idx+flicker_length):
                if not isflicker:
                    break
                # Frames should alternate.
                thisclass = boards[i_idx][row,col]
                if lastclass is flicker_clr:
                    if thisclass is not Puyo.NONE:
                        isflicker = False
                elif lastclass is Puyo.NONE:
                    if thisclass is not flicker_clr:
                        isflicker = False
                lastclass = thisclass
            # If the puyo is flickering, check to see if it is later than the latest flicker and log it.
            if isflicker:
                if idx > (frameno+flicker_length+10): # this is the latest flicker (padded), make a new set
                    frameno = idx
                    flicker = [(row,col,flicker_clr)]
                elif idx < frameno: # this flicker is earlier
                    continue
                elif idx == frameno and flicker is not None: # this flicker is at the same time, add to the set
                    flicker.append((row,col,flicker_clr))
                    
    return frameno, flicker, garbage

def reconstructPop(boardafter,pop,gbg=None):
    # Given the pop locations (and color), reconstruct the board pre-pop. TODO: garbage, hidden row
    boardbefore = Puyo.NONE * np.ones_like(boardafter)
    for c_idx, col in enumerate(boardafter.T):
        r_offset = 0
        col_pop = [x for x in pop if x[1] == c_idx]
        col_pop = sorted(col_pop, key=lambda x: x[0])
        for r_idx, puyo in enumerate(col):
            if not col_pop or col_pop[0][0] > r_idx:
                boardbefore[r_idx,c_idx] = boardafter[r_idx-r_offset,c_idx]
            else:
                boardbefore[r_idx,c_idx] = col_pop.pop(0)[2]
                r_offset += 1
    return boardbefore

def processClf(clf):
    # This function will process a list of frame classifications into a puyo placement sequence, assuming
    # the sequence begins at the start of the game and end at the end. TODO: Identify start/end.
    # 1) Frame classifications are stabilized by a greedy filter which requires some number of frames (11)
    #    following the current frame to have the same classification prior to registering the class change.
    #    TODO: Verify for garbage.
    # 2) Next puyo transition frames are identified for evaluating where the last puyo was placed. In most
    #    cases, the new board is a strict addition to the old board (with garbage as well, potentially).
    # 3) TODO: If both puyo placements are not witnessed, then the puyo was placed in the hidden row.
    # 4) If the new board is not a strict addition then a pop sequence occurred and requires further
    #    detailed processing on the unstabilized frame classifications.
    #      a. In reverse order from the final board, frames are searched for flash-sequences (10 frames
    #         of alternating classification between a puyo and empty). The chain is then reconstructed
    #         in reverse order; puyos thats were in the hidden row may be identified along with the
    #         previous puyo placement.
    #       TODO: It is possible that board vibration may disrupt the flash-sequence search. This has
    #             not yet been witnessed. Garbage will dissapear on the first flash.

    stable_clf = flashStabilize(clf,6)
    transition_frames = getTransitionFrames(clf)
    transition_boards = getTransitionBoards(stable_clf,transition_frames) # TODO: Convert previous code to numpy.
    transitions = list(zip(transition_frames,transition_boards))

    lastb = None
    lastf = None
    popped = False
    popboards = []
    for f,b in transitions:
        if lastb is not None:
            popped = boardPopped(b,lastb)
        if popped:
            fno = f
            pop = True
            boardbefore = b
            while pop:
                fno, pop, gbg = findLastFlicker(clf,(lastf,fno))
                if pop is None: break
                boardbefore = reconstructPop(boardbefore,pop)
                popboards.append((fno,boardbefore))
        lastb = b
        lastf = f

    transitions = sorted(transitions + popboards, key=lambda x: x[0])

    # for _,b in transitions:
    #     img = puyodebug.plotBoardState(b)
    #     cv2.imshow('',img)
    #     cv2.waitKey(0)
        
    return transitions

#path = "./test_data/MvT_58758_60621/"
path = "./test_data/MvT_96000_98129/"
clf = pickle.load(open(path + "clf_list.p","rb"))
framelist = pickle.load(open(path + "frame_list.p","rb"))

# transitions = processClf(clf)
from puyorc import robustClassify
clf = clf[3:-80]
framelist = framelist[3:-80]
board_seq = robustClassify(clf)
#print(len(framelist),len(clf))
puyodebug.makeMovie(framelist,board_seq,clf)

#for _,board in board_seq:
#    cv2.imshow('',puyodebug.plotBoardState(board))
#    cv2.waitKey(0)


#stable_clf = flashStabilize(clf,6)
#plotClf(clf,row=9)
#plotClf(clf,row=8,stable_clf=stable_clf)
#plotClf(clf,row=3,stable_clf=stable_clf)
#plotClf(clf,row=2,stable_clf=stable_clf)
#plotClf(clf,row=5,stable_clf=stable_clf)
#plotClf(clf,row=6,stable_clf=stable_clf)
#plt.show()

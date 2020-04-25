from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
from matplotlib import gridspec
from puyocv import Puyo
import numpy as np
import cv2

_PUYO_COLORS = {Puyo.RED: 'red',
                Puyo.YELLOW: 'goldenrod',
                Puyo.GREEN: 'forestgreen',
                Puyo.BLUE: 'royalblue',
                Puyo.PURPLE: 'darkviolet',
                Puyo.GARBAGE: 'grey',
                Puyo.NONE: 'black'}

def plotBoardState(board): # TODO: Add hidden row input and plotting.
    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex='col',
                                  gridspec_kw={'height_ratios':[1,12], 'hspace':0},
                                  figsize=(2,4), dpi=80)

    if board is not None:
        for ix,iy in np.ndindex(board.shape):
            if board[ix,iy] is not Puyo.NONE:
                ax2.plot(iy+0.5,ix+0.5,'o',markersize=11,color=_PUYO_COLORS[board[ix,iy]])

    plt.setp(ax2, xticks=range(0,7), xticklabels='', yticks=range(0,13), yticklabels='')
    ax2.set_xlim([0,6])
    ax2.set_ylim([0,12])

    plt.setp(ax1, xticks=range(0,7), xticklabels='', yticks=range(0,2), yticklabels='')   
    ax1.set_xlim([0,6])
    ax1.set_ylim([0,1])

    ax1.grid(b=True,which='major',linestyle='-',linewidth=0.5)
    ax2.grid(b=True,which='major',linestyle='-',linewidth=0.5)
    
    for axi in (ax2.xaxis, ax2.yaxis, ax1.xaxis, ax1.yaxis):
        for tick in axi.get_major_ticks():
            tick.tick1line.set_visible(False)

    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape((80*4,80*2,3))
    image = image[32:-28,15:-11]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return image

def plotVideoOverlay(clf,boardframe,nextframe):
    fig, ax = plt.subplots(figsize=(4,8),dpi=80)
    boardframe = cv2.cvtColor(boardframe, cv2.COLOR_BGR2RGB)
    ax.imshow(boardframe, extent=[0.5,6.5,0.5,12.5])
    nextframe = cv2.cvtColor(nextframe, cv2.COLOR_BGR2RGB)
    ax.imshow(nextframe, extent=[6.5,7.5,10.5,12.5])
    for row in range(1,13):
        for col in range(1,8):
            if col < 7:
                res = clf[0][row-1][col-1]
            elif col == 7 and (row == 11 or row == 12):
                res = clf[1][row-11]
            else:
                continue
            ax.plot(col, row, 'o', markersize=12, color=_PUYO_COLORS[res])
    plt.axis('off')
    canvas = FigureCanvas(fig)
    canvas.draw()
    overlay = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape((80*8,80*4,3))
    overlay = overlay[100:-100,:]
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return overlay

def mergeImages(board,overlay):
    height1, width1, _ = overlay.shape
    height2, width2, _ = board.shape
    merged = 255 * np.ones((height1,width1+width2,3),dtype=np.uint8)
    merged[0:height1,0:width1] = overlay
    merged[height1-height2-1:-1,width1-61:-61] = board
    merged = merged[:,35:-50]
    return merged

def makeMovie(clf,transitions,framelist):
    # transitions is an array of tuples (frame_no, board_state)
    # boardframes is the raw image of the player board
    # nextframes is the raw images of the players next puyo
    video = None
    tidx = 0
    board_state = plotBoardState(None)
    for idx, (boardframe,nextframe) in enumerate(framelist):
        if tidx < len(transitions) and idx == transitions[tidx][0]:
            board_state = plotBoardState(transitions[tidx][1])
            tidx += 1
        overlay = plotVideoOverlay(clf[idx],boardframe,nextframe)
        merged = mergeImages(board_state,overlay)

        if video is None:
            h,w,_ = merged.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter('puyo.mp4',fourcc,30,(w,h))
        
        video.write(merged)

    cv2.destroyAllWindows()
    video.release()

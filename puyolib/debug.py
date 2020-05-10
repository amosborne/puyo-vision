from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
from collections import namedtuple
from puyolib.puyo import Puyo
from puyolib.robustify import robustClassify
import puyolib.vision
import numpy as np
import cv2
import multiprocessing as mp

_PUYO_COLORS = {
    Puyo.RED: "red",
    Puyo.YELLOW: "goldenrod",
    Puyo.GREEN: "forestgreen",
    Puyo.BLUE: "royalblue",
    Puyo.PURPLE: "darkviolet",
    Puyo.GARBAGE: "grey",
    Puyo.NONE: "black",
}


def plotBoardState(board):  # TODO: Add hidden row input and plotting.
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex="col",
        gridspec_kw={"height_ratios": [1, 12], "hspace": 0},
        figsize=(2, 4),
        dpi=80,
    )

    if board is not None:
        for ix, iy in np.ndindex(board.shape):
            if board[ix, iy] is not Puyo.NONE:
                ax2.plot(
                    iy + 0.5,
                    ix + 0.5,
                    "o",
                    markersize=11,
                    color=_PUYO_COLORS[board[ix, iy]],
                )

    plt.setp(
        ax2, xticks=range(0, 7), xticklabels="", yticks=range(0, 13), yticklabels=""
    )
    ax2.set_xlim([0, 6])
    ax2.set_ylim([0, 12])

    plt.setp(
        ax1, xticks=range(0, 7), xticklabels="", yticks=range(0, 2), yticklabels=""
    )
    ax1.set_xlim([0, 6])
    ax1.set_ylim([0, 1])

    ax1.grid(b=True, which="major", linestyle="-", linewidth=0.5)
    ax2.grid(b=True, which="major", linestyle="-", linewidth=0.5)

    for axi in (ax2.xaxis, ax2.yaxis, ax1.xaxis, ax1.yaxis):
        for tick in axi.get_major_ticks():
            tick.tick1line.set_visible(False)

    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
        (80 * 4, 80 * 2, 3)
    )
    image = image[32:-28, 15:-11]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return image


def getSubImage(frame, dim):
    top, left, height, width = dim
    subimg = frame[top : top + height, left : left + width]
    subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2RGB)
    subimg = np.uint8(subimg)
    return subimg


def plotVideoOverlay(frame, player, clf):
    fig, ax = plt.subplots(figsize=(4, 8), dpi=80)
    # Get the appropriate sub-images depending on the player.
    if player == 1:
        board_dim = puyolib.vision.P1_BOARD_DIMENSION
        nextpuyo_dim = puyolib.vision.P1_NEXT_DIMENSION
    elif player == 2:
        board_dim = puyolib.vision.P2_BOARD_DIMENSION
        nextpuyo_dim = puyolib.vision.P2_NEXT_DIMENSION
    boardframe = getSubImage(frame, board_dim)
    ax.imshow(boardframe, extent=[0.5, 6.5, 0.5, 12.5])
    nextframe = getSubImage(frame, nextpuyo_dim)
    ax.imshow(nextframe, extent=[6.5, 7.5, 10.5, 12.5])
    # Plot colored dots on top to show the classification.
    for row in range(1, 13):
        for col in range(1, 8):
            if col < 7:
                res = clf[0][row - 1][col - 1]
            elif col == 7 and (row == 11 or row == 12):
                res = clf[1][row - 11]
            else:
                continue
            ax.plot(col, row, "o", markersize=12, color=_PUYO_COLORS[res])
    plt.axis("off")
    # Export in BGR format for OpenCV handling.
    canvas = FigureCanvas(fig)
    canvas.draw()
    overlay = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
        (80 * 8, 80 * 4, 3)
    )
    overlay = overlay[100:-100, :]
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return overlay


def mergeImages(board, overlay):
    height1, width1, _ = overlay.shape
    height2, width2, _ = board.shape
    merged = 255 * np.ones((height1, width1 + width2, 3), dtype=np.uint8)
    merged[0:height1, 0:width1] = overlay
    merged[height1 - height2 - 1 : -1, width1 - 61 : -61] = board
    merged = merged[:, 35:-50]
    return merged


FrameData = namedtuple("FrameData", ["frame", "p1clf", "p2clf", "p1board", "p2board"])


def createMovieFrame(frame_data):
    p1board_state = plotBoardState(frame_data.p1board)
    p1overlay = plotVideoOverlay(frame_data.frame, 1, frame_data.p1clf)
    p1merged = mergeImages(p1board_state, p1overlay)
    p2board_state = plotBoardState(frame_data.p2board)
    p2overlay = plotVideoOverlay(frame_data.frame, 2, frame_data.p2clf)
    p2merged = mergeImages(p2board_state, p2overlay)
    return np.hstack((p1merged, p2merged))


def frameDataGenerator(cap, record):
    p1board = p2board = None
    p1board_seq = robustClassify(record.p1clf)
    p2board_seq = robustClassify(record.p2clf)
    for fno, (p1clf, p2clf) in enumerate(zip(record.p1clf, record.p2clf)):
        ret, frame = cap.read()
        if not ret:
            break
        if p1board_seq and fno >= p1board_seq[0].frameno - 1:
            p1board = p1board_seq.pop(0).board
        if p2board_seq and fno >= p2board_seq[0].frameno - 1:
            p2board = p2board_seq.pop(0).board
        yield FrameData(
            frame=frame, p1clf=p1clf, p2clf=p2clf, p1board=p1board, p2board=p2board
        )


def makeMovie(dest, src, record):

    # Write the video.
    video = None
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, record.start_frame)
    pool = mp.Pool(puyolib.vision.CPU_COUNT)
    for image in pool.imap(createMovieFrame, frameDataGenerator(cap, record)):
        if video is None:
            h, w, _ = image.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(dest + ".mp4", fourcc, 30, (w, h))
        video.write(image)

    pool.terminate()
    video.release()
    cap.release()

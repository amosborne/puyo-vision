from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
from matplotlib import gridspec
from puyolib.puyo import Puyo
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


def createMovieFrame(frame_data):
    frame, player, clf, board = frame_data
    board_state = plotBoardState(board)
    overlay = plotVideoOverlay(frame, player, clf)
    merged = mergeImages(board_state, overlay)
    return merged


def frameDataGenerator(cap, player, board_seq, clf_list):
    board = None
    for fno, clf in enumerate(clf_list):
        ret, frame = cap.read()
        if not ret:
            break
        if board_seq and fno >= board_seq[0].frameno:
            board = board_seq.pop(0).board
        yield (frame, player, clf, board)


def makeMovie(dest, src, player, board_seq, record):

    # Unpack the game record.
    start_frame = record[0][2]
    if player == 1:
        clf_list = record[1]
    elif player == 2:
        clf_list = record[2]

    # Write the video.
    video = None
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pool = mp.Pool(puyolib.vision.CPU_COUNT)
    for image in pool.imap(
        createMovieFrame, frameDataGenerator(cap, player, board_seq, clf_list)
    ):
        if video is None:
            h, w, _ = image.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(dest + ".mp4", fourcc, 30, (w, h))
        video.write(image)

    pool.terminate()
    video.release()
    cap.release()

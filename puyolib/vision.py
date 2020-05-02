import os
from inspect import getsourcefile
from copy import deepcopy


from csv import reader as csvreader
from enum import IntEnum, auto
from collections import defaultdict, namedtuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pickle

# README
# 1) Assumed  game: Puyo Puyo Champions / Esports
# 2) Assumed video: fullscreen 720p

# Path to the SVM training data.
_TRAINING_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(getsourcefile(lambda: 0))), "training_data/"
)

P1_PIXEL_WINDOW = (109, 585, 185, 443)  # manually tuned
P1_PIXEL_NEXT_WINDOW = (107, 191, 478, 524)  # manually tuned
P2_PIXEL_WINDOW = (109, 585, 834, 1094)  # manually tuned
P2_PIXEL_NEXT_WINDOW = (107, 191, 755, 801)  # manually tuned

SCREEN_RESOLUTION = (720, 1280)  # (height, weidth)
P1_BOARD_DIMENSION = (109, 186, 476, 258)  # (top, left, heigh, width)


class Puyo(IntEnum):
    NONE = 0
    RED = auto()
    YELLOW = auto()
    GREEN = auto()
    BLUE = auto()
    PURPLE = auto()
    GARBAGE = auto()


def enableOCL():
    """Enables OpenCL usage if a device is available."""

    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
    return cv2.ocl.useOpenCL()


def loadTrainingData():
    """Create a dictionary of puyo training data."""

    train_data_list = []
    training_files = os.listdir(_TRAINING_DATA_PATH)
    for filename in training_files:
        # Find the files with the appropriate filenames.
        if filename.endswith(".jpg"):
            rootname = filename.split(".")[0]
            dataname = rootname + "_data.dat"
            if dataname not in training_files:
                continue
        else:
            continue
        # Load the data into a training dictionary.
        img = cv2.imread(_TRAINING_DATA_PATH + filename)
        img = cv2.UMat(img)
        p1dict, p2dict = readTrainingDataCSV(_TRAINING_DATA_PATH + dataname)
        train_data_list.append((img, p1dict, p2dict))
    return train_data_list


def readTrainingDataCSV(filename):
    """Read in the puyo coordinates from the training data CSV's."""

    p1_pos = {}
    p2_pos = {}
    with open(filename) as csvfile:
        reader = csvreader(csvfile, delimiter=" ")
        for line in reader:
            # Data is in (row,col) pairs per player_color.
            dataline = [x for x in filter(None, line)]
            player, color = dataline.pop(0).split("_")
            dataline = [tuple(map(int, x.split(","))) for x in dataline]
            if player == "P1":
                p1_pos[Puyo[color]] = dataline
            elif player == "P2":
                p2_pos[Puyo[color]] = dataline
    return p1_pos, p2_pos


def trainClassifier():
    """Train a new group of SVM's for all-vs-one HOG classification."""

    # Load the training data.
    train_data_list = loadTrainingData()

    # Create the HOG operator.
    winSize = (48, 48)
    blockSize = (12, 12)
    blockStride = (6, 6)
    cellSize = (6, 6)
    nbins = 5
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    # Define the second board dimension symetrically off the first.
    top, left1, height, width = P1_BOARD_DIMENSION
    left2 = SCREEN_RESOLUTION[1] - (left1 + width)
    P2_BOARD_DIMENSION = (top, left2, height, width)
    padding = 0

    # Extract the hog features from the training set.
    features = defaultdict(list)
    for img, p1dict, p2dict in train_data_list:
        for puyo_type, pos_list in p1dict.items():
            for pos in pos_list:
                puyo_img = getPuyoImage(img, P1_BOARD_DIMENSION, pos, winSize, padding)
                features[puyo_type].append(hog.compute(puyo_img))
        for puyo_type, pos_list in p2dict.items():
            for pos in pos_list:
                puyo_img = getPuyoImage(img, P2_BOARD_DIMENSION, pos, winSize, padding)
                features[puyo_type].append(hog.compute(puyo_img))

    # Create hog classifiers.
    svm = generateClassifier(features)

    for img, p1dict, p2dict in train_data_list:
        for puyo_type, pos_list in p1dict.items():
            for pos in pos_list:
                puyo_img = getPuyoImage(img, P1_BOARD_DIMENSION, pos, winSize, padding)
                print(puyo_img.get().shape)
                cv2.imshow("", puyo_img)
                feature = np.array(hog.compute(puyo_img), dtype=np.float32)
                res = svm.predict(np.transpose(feature))
                print(Puyo(res[1][0]))
                cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None


def generateClassifier(features):
    """Return all vs. one hog classifiers for each puyo type."""

    svm = cv2.ml.SVM_create()
    svm_features = []
    svm_responses = []
    # for feature_type, feature_list in features.items():
    for puyo_type in Puyo:
        feature_type = puyo_type
        feature_list = features[feature_type]
        svm_features += feature_list
        svm_responses += [feature_type] * len(feature_list)
    svm_features = np.array(svm_features, dtype=np.float32)
    svm_responses = np.array(svm_responses, dtype=np.int32)
    svm.trainAuto(svm_features, cv2.ml.ROW_SAMPLE, svm_responses)
    return svm


def getPuyoImage(img, dimension, pos, size, pad):
    """Return the properly sized puyo image given the full frame image."""

    top, left, height, width = dimension
    row, col = pos
    puyo_center = (
        round(top + height - (height / 12) * (row - 0.5)),
        round(left + (width / 6) * (col - 0.5)),
    )
    puyo_img = cv2.UMat(
        img,
        [puyo_center[0] - size[0] // 2 - pad, puyo_center[0] + size[0] // 2 + pad],
        [puyo_center[1] - size[1] // 2 - pad, puyo_center[1] + size[1] // 2 + pad],
    )
    puyo_img = cv2.cvtColor(puyo_img, cv2.COLOR_BGR2GRAY)
    return puyo_img


def getNextPuyoImage(image, corners, pos):
    # sz = PUYO_PIXEL_SIZE // 2
    # row, col = pos
    # y = round(corners[1] - ((row - 0.5) * (corners[1] - corners[0]) / 2))
    # x = round(corners[2] + ((col - 0.5) * (corners[3] - corners[2])))
    # subimage = image[y - sz : y + sz, x - sz : x + sz]
    return None


def trainSVM(data):
    # features = []
    # responses = []
    # for puyo, image_list in data.items():
    #     for image in image_list:
    #         gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         features.append(HOG.compute(gimage))
    #         responses.append(puyo)
    # features = np.array(features, dtype=np.float32)
    # responses = np.array(responses, dtype=np.int32)
    # svm = cv2.ml.SVM_create()
    # svm.trainAuto(features, cv2.ml.ROW_SAMPLE, responses)
    return None


# Train a multi-class SVM using HOG with hand-crafted training data.
SVM = trainSVM(loadTrainingData())


def shakeAdjust(window, shake):
    return (window[0], window[1], window[2] + shake, window[3] + shake)


def predictPuyo(image, player, pos, nextpuyo, shake):
    if player == 1 and not nextpuyo:
        win = shakeAdjust(P1_PIXEL_WINDOW, shake)
        puyo = getPuyoImage(image, win, pos)
    elif player == 2 and not nextpuyo:
        win = shakeAdjust(P2_PIXEL_WINDOW, shake)
        puyo = getPuyoImage(image, win, pos)
    elif player == 1 and nextpuyo:
        puyo = getNextPuyoImage(image, P1_PIXEL_NEXT_WINDOW, pos)
    elif player == 2 and nextpuyo:
        puyo = getNextPuyoImage(image, P2_PIXEL_NEXT_WINDOW, pos)
    gpuyo = cv2.cvtColor(puyo, cv2.COLOR_BGR2GRAY)
    # feature = HOG.compute(gpuyo)
    # feature = np.array(feature, dtype=np.float32)
    # feature = np.transpose(feature)
    # result = SVM.predict(feature)
    # result = Puyo(result[1][0])
    return None


def classifyFrame(frame, player, shake):
    # Given a frame, will make a classification on every board position and the next puyo for the player.
    board = np.empty([12, 6], dtype=Puyo)
    for row in range(1, 13):
        for col in range(1, 7):
            res = predictPuyo(frame, player, (row, col), False, shake)
            board[row - 1][col - 1] = res
    n1 = predictPuyo(frame, player, (1, 1), True, 0)
    n2 = predictPuyo(frame, player, (2, 1), True, 0)
    return (board, (n1, n2))


def getSubImage(image, window):
    subimage = image[window[0] : window[1], window[2] : window[3]]
    return subimage


def getPlayerSubFrames(frame, player, shake):
    if player == 1:
        win = shakeAdjust(P1_PIXEL_WINDOW, shake)
        bimg = getSubImage(frame, win)
        nimg = getSubImage(frame, P1_PIXEL_NEXT_WINDOW)
    elif player == 2:
        win = shakeAdjust(P2_PIXEL_WINDOW, shake)
        bimg = getSubImage(frame, win)
        nimg = getSubImage(frame, P2_PIXEL_NEXT_WINDOW)
    return bimg, nimg


def drawClf(frame, player, clf, shake):
    # Given a frame and a classification, will draw the classification on top of the frame for the player.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if player == 1:
        win = shakeAdjust(P1_PIXEL_WINDOW, shake)
        bimg = getSubImage(frame, win)
        nimg = getSubImage(frame, P1_PIXEL_NEXT_WINDOW)
    elif player == 2:
        win = shakeAdjust(P2_PIXEL_WINDOW, shake)
        bimg = getSubImage(frame, win)
        nimg = getSubImage(frame, P2_PIXEL_NEXT_WINDOW)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=80)
    ax.imshow(bimg, extent=[0.5, 6.5, 0.5, 12.5])
    ax.imshow(nimg, extent=[6.5, 7.5, 10.5, 12.5])
    for row in range(1, 13):
        for col in range(1, 8):
            if col < 7:
                res = clf[0][row - 1][col - 1]
            elif col == 7 and (row == 11 or row == 12):
                res = clf[1][row - 11]
            else:
                continue
            if res == Puyo.RED:
                clr = "red"
            elif res == Puyo.YELLOW:
                clr = "goldenrod"
            elif res == Puyo.GREEN:
                clr = "forestgreen"
            elif res == Puyo.BLUE:
                clr = "royalblue"
            elif res == Puyo.PURPLE:
                clr = "darkviolet"
            elif res == Puyo.GARBAGE:
                clr = "grey"
            elif res == Puyo.NONE:
                clr = "black"
            ax.plot(col, row, "o", markersize=12, color=clr)
    plt.axis("off")
    canvas = FigureCanvas(fig)
    canvas.draw()
    overlay = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(frame.shape)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    overlay = overlay[65:-65, 480:-445]  # magic numbers
    plt.close(fig)
    return overlay


def expandWindow(win, pix):
    return (win[0], win[1], win[2] - pix, win[3] + pix)


def calcShake(thisframe, lastframe, player, prevshake):
    if player == 1:
        win = shakeAdjust(P1_PIXEL_WINDOW, prevshake)
    elif player == 2:
        win = shakeAdjust(P2_PIXEL_WINDOW, prevshake)
    win = expandWindow(win, -25)  # magic number
    src1 = getSubImage(thisframe, win)
    src1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
    src1 = np.float32(src1)
    src2 = getSubImage(lastframe, win)
    src2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    src2 = np.float32(src2)
    (xadj, _), _ = cv2.phaseCorrelate(src1, src2)
    xadj = int(xadj)
    if xadj == 0 and not prevshake == 0:
        shake = 0
    else:
        shake = prevshake - xadj
    if abs(shake) > 100:  # magic number
        shake = 0
    return shake


def main():
    enableOCL()
    trainClassifier()
    return None

    # Do an informal test.
    cap = cv2.VideoCapture("momoken_vs_tom.mp4")
    end_frame = 98129
    start_frame = 96000

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # while True:
    #     cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    #     _, thisframe = cap.read()
    #     cv2.imshow('',thisframe)
    #     press = cv2.waitKey(1)
    #     if press & 0xFF == ord('q'):
    #         print(cur_frame)
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    # return

    framelist = []
    clflist = []

    shake = 0
    lastframe = None
    while True:
        # If the current frame is the last frame needed, break.
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == end_frame:
            break
        # Read the current frame.
        _, thisframe = cap.read()
        # If the last frame is not none, calculate the screen shake for player 1.
        if lastframe is not None:
            shake = calcShake(thisframe, lastframe, 1, shake)
        # Classify (and draw) the screen with shake accounted for.
        clf = classifyFrame(thisframe, 1, shake)
        img = drawClf(thisframe, 1, clf, shake)
        # Write the frames and the clf to lists to pickle afterwards.
        framelist.append(getPlayerSubFrames(thisframe, 1, shake))
        clflist.append(clf)

        # Press 'q' to quit early. Display the overlayed classification.
        cv2.imshow("", img)
        press = cv2.waitKey(1)
        if press & 0xFF == ord("q"):
            break

        lastframe = thisframe

    cap.release()
    cv2.destroyAllWindows()

    pickle.dump(framelist, open("frame_list.p", "wb"))
    pickle.dump(clflist, open("clf_list.p", "wb"))


if __name__ == "__main__":
    main()
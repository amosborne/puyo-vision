import os
from inspect import getsourcefile
from csv import reader as csvreader
from collections import defaultdict
import cv2
import numpy as np
from puyolib.puyo import Puyo
import puyolib.debug


def enableOCL(b=True):
    """Enables OpenCL usage if a device is available."""

    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(b)
    return cv2.ocl.useOpenCL()


def loadTrainingData():
    """Create a dictionary of puyo training data."""

    train_data_list = []
    training_files = os.listdir(TRAINING_DATA_PATH)
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
        img = cv2.imread(TRAINING_DATA_PATH + filename)
        img = cv2.UMat(img)
        p1dict, p2dict = readTrainingDataCSV(TRAINING_DATA_PATH + dataname)
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

    # Check if the SVM already exists.
    if os.path.exists(TRAINING_DATA_PATH + SVM_FILENAME):
        return cv2.ml.SVM_load(TRAINING_DATA_PATH + SVM_FILENAME)

    # Load the training data.
    train_data_list = loadTrainingData()

    # Extract the hog features from the training set. Training images are
    # on the main board without any ongoing shake animation.
    features = defaultdict(list)
    for img, p1dict, p2dict in train_data_list:
        for puyo_type, pos_list in p1dict.items():
            for pos in pos_list:
                puyo_img = getPuyoImage(img, P1_BOARD_DIMENSION, pos, winSize)
                features[puyo_type].append(HOG.compute(puyo_img))
        for puyo_type, pos_list in p2dict.items():
            for pos in pos_list:
                puyo_img = getPuyoImage(img, P2_BOARD_DIMENSION, pos, winSize)
                features[puyo_type].append(HOG.compute(puyo_img))

    # Return SVM classifier.
    return createSVMClassifier(features)


def createSVMClassifier(features):
    """Return multiclass SVM classifier covering all puyo types."""

    svm = cv2.ml.SVM_create()
    svm_features = []
    svm_responses = []
    for feature_type, feature_list in features.items():
        svm_features += feature_list
        svm_responses += [feature_type] * len(feature_list)
    svm_features = np.array(svm_features, dtype=np.float32)
    svm_responses = np.array(svm_responses, dtype=np.int32)
    svm.trainAuto(svm_features, cv2.ml.ROW_SAMPLE, svm_responses)
    svm.save(TRAINING_DATA_PATH + SVM_FILENAME)
    return svm


def getPuyoImage(img, dimension, pos, size, nextpuyo=False):
    """Return the properly sized puyo image given the full frame image."""

    top, left, height, width = dimension
    row, col = pos
    if nextpuyo:
        puyo_center = (
            round(top + height - (height / 2) * (row - 0.5)),
            round(left + (width / 1) * (col - 0.5)),
        )
    else:
        puyo_center = (
            round(top + height - (height / 12) * (row - 0.5)),
            round(left + (width / 6) * (col - 0.5)),
        )
    puyo_img = cv2.UMat(
        img,
        [puyo_center[0] - size[0] // 2, puyo_center[0] + size[0] // 2],
        [puyo_center[1] - size[1] // 2, puyo_center[1] + size[1] // 2],
    )
    puyo_img = cv2.cvtColor(puyo_img, cv2.COLOR_BGR2GRAY)
    puyo_img = puyo_img.get()  # Image retrieved from GPU due to HOG constraints.
    return puyo_img


def predictPuyo(frame, player, pos, nextpuyo=False, shake=0):
    """Return the classification of the puyo on the board or in the next window."""

    if nextpuyo:
        if player == 1:
            dim = P1_NEXT_DIMENSION
        elif player == 2:
            dim = P2_NEXT_DIMENSION
    else:
        if player == 1:
            dim = P1_BOARD_DIMENSION
        elif player == 2:
            dim = P2_BOARD_DIMENSION
        dim = shakeAdjust(dim, shake)
    puyo_img = getPuyoImage(frame, dim, pos, winSize, nextpuyo)
    feature = HOG.compute(puyo_img)
    return Puyo(SVM.predict(np.transpose(feature))[1][0])


def classifyFrame(frame, player):
    """Classify each board position and next window frame for the given player."""

    board = np.empty([12, 6], dtype=Puyo)
    for row in range(1, 13):
        for col in range(1, 7):
            res = predictPuyo(frame, player, (row, col), shake=0)  # update
            board[row - 1][col - 1] = res
    n1 = predictPuyo(frame, player, (1, 1), nextpuyo=True)
    n2 = predictPuyo(frame, player, (2, 1), nextpuyo=True)
    return (board, (n1, n2))


def shakeAdjust(dimension, shake):
    """Alter the given dimension to horizontal shake."""

    top, left, height, width = dimension
    return (top, left + shake, height, width)


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


def bySymmetry(p1_dim, res):
    """Define the player 2 dimension symmetrically off the first."""

    top, left1, height, width = p1_dim
    left2 = res[1] - (left1 + width)
    return (top, left2, height, width)


# Path to the SVM training data.
SVM_FILENAME = "svm.svm"
TRAINING_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(getsourcefile(lambda: 0))), "training_data/"
)

# Create the HOG operator and SVM classifier.
winSize = (48, 48)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 5
HOG = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
SVM = trainClassifier()

# Define fixed screen pixel dimensions.
SCREEN_RESOLUTION = (720, 1280)  # (height, weidth)
P1_BOARD_DIMENSION = (109, 186, 476, 258)  # (top, left, height, width)
P1_NEXT_DIMENSION = (107, 479, 84, 46)  # (top, left, height, width)
P2_BOARD_DIMENSION = bySymmetry(P1_BOARD_DIMENSION, SCREEN_RESOLUTION)
P2_NEXT_DIMENSION = bySymmetry(P1_NEXT_DIMENSION, SCREEN_RESOLUTION)


def main():
    enableOCL()
    frame = cv2.UMat(cv2.imread("./puyolib/training_data/image2.jpg"))
    clf = classifyFrame(frame, 1)
    overlay = puyolib.debug.plotVideoOverlay(
        clf, frame, P1_BOARD_DIMENSION, P1_NEXT_DIMENSION
    )
    cv2.imshow("", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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

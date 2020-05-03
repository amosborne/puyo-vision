import os
from inspect import getsourcefile
from csv import reader as csvreader
from collections import defaultdict
import cv2
import numpy as np
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity as ssim
from puyolib.puyo import Puyo


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
            shk = trackBoardEdge(frame, player)
            if shk is None:
                return None
            res = predictPuyo(frame, player, (row, col), shake=shk)
            board[row - 1][col - 1] = res
    n1 = predictPuyo(frame, player, (1, 1), nextpuyo=True)
    n2 = predictPuyo(frame, player, (2, 1), nextpuyo=True)
    return (board, (n1, n2))


def shakeAdjust(dimension, shake):
    """Alter the given dimension to horizontal shake."""

    top, left, height, width = dimension
    return (top, left + shake, height, width)


def trackBoardEdge(frame, player):
    """Extract the board edge for shake and end-game detection."""

    if player == 1:
        dim = P1_EDGE_WINDOW
        offset = P1_EDGE_OFFSET
        idx = 0
    elif player == 2:
        dim = P2_EDGE_WINDOW
        offset = P2_EDGE_OFFSET
        idx = -1
    # Apply a 5 pixel square blur after thresholding to near-white.
    top, left, height, width = dim
    raw_img = cv2.UMat(frame, [top, top + height], [left, left + width])
    gry_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    _, grt_img = cv2.threshold(gry_img, 200, 255, cv2.THRESH_BINARY)
    blur_img = cv2.blur(grt_img, (5, 5))
    # Find the average horizonal position of the vertical edge.
    blur_data = np.divide(blur_img.get(), 255)
    xs = []
    for row in blur_data:
        peaks, _ = find_peaks(row, height=0.25)
        if len(peaks) == 0:
            return None
        else:
            xs.append(peaks[idx])
    return int(np.average(xs)) - offset


def bySymmetry(p1_dim, res):
    """Define the player 2 dimension symmetrically off the first."""

    top, left1, height, width = p1_dim
    left2 = res[1] - (left1 + width)
    return (top, left2, height, width)


def scoreImageProcess(img):
    """Turn the score to grayscale and threshold."""

    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(new_img, 235, 255, cv2.THRESH_BINARY)[1]


def isGameStart(frame):
    """Identify the beginning of a game using SSIM."""

    top, left, height, width = P1_START_SCORE
    score_img = cv2.UMat(frame, [top, top + height], [left, left + width])
    score_img = scoreImageProcess(score_img).get()
    if ssim(score_img, START_IMAGE) > 0.75:
        return True
    return False


def processVideo(filepath, start_frameno=0, end_frameno=None, ngames=None):
    """Return a list of frame-by-frame classifications for a video.
    
    Each element of the output list is a tuple:
    ( (start_frameno, end_frameno), [player1_clfs], [player2_clfs] )
    """

    # Initialize the video capture stream.
    cap = cv2.VideoCapture(filepath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frameno)

    game_record = []
    ingame = False
    while True:
        # If the current frame is the last frame needed, break.
        if end_frameno is not None and cap.get(cv2.CAP_PROP_POS_FRAMES) == end_frameno:
            break

        # Read the current frame. Break if at the end.
        ret, cpu_frame = cap.read()
        if not ret:
            break

        # Check if the video frame is within a game.
        gpu_frame = cv2.UMat(cpu_frame)
        if not ingame:
            if not isGameStart(gpu_frame):
                continue
            else:
                ingame = True
                start_fno = cap.get(cv2.CAP_PROP_POS_FRAMES)
                clf1_list = []
                clf2_list = []

        # If within a game, process until the end.
        if ingame:
            clf1 = classifyFrame(gpu_frame, 1)
            clf2 = classifyFrame(gpu_frame, 2)
            # Check for end game.
            if clf1 is None or clf2 is None:
                ingame = False
                end_fno = cap.get(cv2.CAP_PROP_POS_FRAMES)
                game_record.append(((start_fno, end_fno), clf1_list, clf2_list))
                # Break early if there was a number of games desired.
                if ngames is not None and len(game_record) == ngames:
                    break
            else:
                clf1_list.append(clf1)
                clf2_list.append(clf2)

    # Release the video capture stream.
    cap.release()

    return game_record


# Path to the SVM training data.
SVM_FILENAME = "svm.svm"
ROOT_PATH = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
TRAINING_DATA_PATH = os.path.join(ROOT_PATH, "training_data/")

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

P1_EDGE_WINDOW = (585, 75, 30, 200)  # (top, left, height, width)
P1_EDGE_OFFSET = 96
P2_EDGE_WINDOW = bySymmetry(P1_EDGE_WINDOW, SCREEN_RESOLUTION)
P2_EDGE_OFFSET = 102

P1_START_SCORE = (588, 390, 40, 60)  # (top, left, height, width)
START_IMAGE = scoreImageProcess(cv2.imread(os.path.join(ROOT_PATH, "start.png")))

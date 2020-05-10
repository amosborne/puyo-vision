from puyolib.vision import gameClassifier
from puyolib.robustify import robustClassify
from puyolib.debug import plotBoardState, makeMovie
import pickle
import cv2
import os

RECORD_PATH = "results/"


def pickleGameRecord(src_id, record):
    """Save to file the results of processing the game record."""

    # Create the folder to store these records if necessary.
    game_record_path = os.path.join(RECORD_PATH, src_id)
    if not os.path.isdir(game_record_path):
        os.mkdir(game_record_path)

    # Pickle the record. Return the pickled filename.
    filepath_no_ext = os.path.join(game_record_path, record.start_time)
    pickle_path = filepath_no_ext + ".p"
    pickle.dump(record, open(pickle_path, "wb"))
    return pickle_path


def unpickleGameRecord(pickle_path):
    return pickle.load(open(pickle_path, "rb"))


def gameRecordVideo(pickle_path, movie_src):
    record = unpickleGameRecord(pickle_path)
    base_filename = pickle_path.split(".")[0]
    makeMovie(base_filename, movie_src, record)


def reviewGameRecord(filepath, player):
    """View board by board robust classification result."""

    record = unpickleGameRecord(filepath)
    if player == 1:
        clf = record.p1clf
    elif player == 2:
        clf = record.p2clf
    board_seq = robustClassify(clf)
    for _, board in board_seq:
        img = plotBoardState(board)
        cv2.imshow("", img)
        press = cv2.waitKey(0)
        if press & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    return None


vid_filepath = ".tmp/momoken_vs_tom2.mp4"
vid_identifier = "testing_results"

pickle_paths = []
for record in gameClassifier(vid_filepath, ngames=4, start="00:11:30"):
    pickle_paths.append(pickleGameRecord(vid_identifier, record))

for pickle_path in pickle_paths:
    gameRecordVideo(pickle_path, vid_filepath)

# rpath = "./results/testing_results/0:11:30.p"
# player = 1
# reviewGameRecord(rpath, player)

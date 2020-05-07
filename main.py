from puyolib.vision import gameClassifier
from puyolib.robustify import robustClassify
from puyolib.debug import plotBoardState, makeMovie
import pickle
import cv2
import os

RECORD_PATH = "results/"


def processGameRecord(identifier, record, movie=""):
    """Save to file the results of processing the game record.

    Processing currently includes:
        1. Pickling the record for future reference.
        2. Making a movie of both players in the record.
    """

    # Create the folder to store these records if necessary.
    game_record_path = os.path.join(RECORD_PATH, identifier)
    if not os.path.isdir(game_record_path):
        os.mkdir(game_record_path)

    # Pickle the record without the frame images.
    base_filename = os.path.join(game_record_path, record[0][0])
    pickle_name = base_filename + ".p"
    pickle.dump(record, open(pickle_name, "wb"))
    # Make movies if requested.
    # if movie:
    #     raw_movie = (movie, start[0], end[0])
    #     board_seq1 = robustClassify(clf1)
    #     makeMovie(base_filename + "_1", raw_movie, 1, board_seq1, clf1)
    #     print("    Player 1 movie complete.")
    #     board_seq2 = robustClassify(clf2)
    #     makeMovie(base_filename + "_2", raw_movie, 2, board_seq2, clf2)
    #     print("    Player 2 movie complete.")
    return pickle_name


def reviewGameRecord(filepath, player):
    """View board by board robust classification result."""

    record = pickle.load(open(filepath, "rb"))
    (start, end), clf1, clf2 = record
    # print("Start Frame: " + str(start[0]) + " (" + str(start[1]) + "s)")
    if player == 1:
        clf = clf1
    elif player == 2:
        clf = clf2
    board_seq = robustClassify(clf)
    for _, board in board_seq:
        img = plotBoardState(board)
        cv2.imshow("", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None


vid_filepath = ".tmp/momoken_vs_tom2.mp4"
vid_identifier = "testing_results"

for record in gameClassifier(vid_filepath, ngames=1, start="00:00:00"):
    record_path = processGameRecord(vid_identifier, record)

# rpath = "./results/testing_results/0:00:10.p"
# record = pickle.load(open(rpath, "rb"))
# processGameRecord(vid_identifier, record, movie=vid_filepath)
# player = 1
# reviewGameRecord(rpath, player)

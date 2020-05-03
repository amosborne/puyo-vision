import pickle
from puyolib.vision import enableOCL, processVideo
from puyolib.robustify import robustClassify
from puyolib.debug import plotBoardState, makeMovie
import cv2
import os

RECORD_PATH = "results/"


def getGameRecords(filepath, start_frameno=0, end_frameno=None, ngames=None):
    """Output the results of the video processing generator."""

    # Initialize video capture.
    enableOCL()
    cap = cv2.VideoCapture(filepath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frameno)
    game_records = []

    # Retrieve game records from the video.
    for record in processVideo(cap):
        if record is None:
            break
        game_records.append(record)
        record_end_frameno = record[0][1]
        if end_frameno is not None and record_end_frameno >= end_frameno:
            break
        elif ngames is not None and len(game_records) >= ngames:
            break

    # Release video capture and return.
    cap.release()
    return game_records


def processGameRecords(identifier, records, movie=False):
    """Save to file the results of processing the game records.

    Processing currently includes:
        1. Pickling the record for future reference.
        2. Making a move of both players in the record.
    """

    # Create the folder to store these records if necessary.
    game_record_path = os.path.join(RECORD_PATH, identifier)
    if not os.path.isdir(game_record_path):
        os.mkdir(game_record_path)

    pickle_names = []
    for record in records:
        # Unpack the record.
        (start, end), frames, clf1, clf2 = record
        _, start_time = start
        # Pickle the record without the frame images.
        base_filename = os.path.join(game_record_path, str(start_time))
        new_record = ((start, end), clf1, clf2)
        pickle_name = base_filename + ".p"
        pickle.dump(new_record, open(pickle_name, "wb"))
        pickle_names.append(pickle_name)
        # Make movies if requested.
        if movie:
            board_seq1 = robustClassify(clf1)
            makeMovie(base_filename + "_1", frames, 1, board_seq1, clf1)
            board_seq2 = robustClassify(clf2)
            makeMovie(base_filename + "_2", frames, 2, board_seq2, clf2)
    return pickle_names


def reviewGameRecord(filepath, player):
    """View board by board robust classification result."""

    record = pickle.load(open(filepath, "rb"))
    _, clf1, clf2 = record
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


# vid_filepath = "./dev/momoken_vs_tom.mp4"
# vid_identifier = "testing_results"

# records = getGameRecords(vid_filepath, ngames=1)
# record_paths = processGameRecords(vid_identifier, records, movie=False)

rpath = "./results/testing_results/10.p"
player = 2
reviewGameRecord(rpath, player)

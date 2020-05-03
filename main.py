import pickle
from puyolib.vision import enableOCL, processVideo
from puyolib.robustify import robustClassify
from puyolib.debug import plotBoardState
import cv2

# enableOCL()
# filepath = "./dev/momoken_vs_tom.mp4"
# record = processVideo(filepath, ngames=1)
# pickle.dump(record, open("game_record.p", "wb"))

record = pickle.load(open("./dev/test_data/first_game.p", "rb"))
_, clf1, clf2 = record[0]
board_seq = robustClassify(clf1)
for _, board in board_seq:
    board_img = plotBoardState(board)
    cv2.imshow("", board_img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
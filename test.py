import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import puyocv

def getImage():
    cap = cv2.VideoCapture("momoken_vs_tom.mp4")
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * 60 * 30)
    while True:
        suc,frame = cap.read()
        cv2.imshow("frame",frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.imwrite("./tst.jpg", frame)
            break
    cap.release()
    cv2.destroyAllWindows()

def getBoard():
    # Given  an image, crop out the first player's board manually.
    image = cv2.imread("training_image.jpg")
    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped = image[108:586, 184:444]
    return resizeBoard(cropped)

def resizeBoard(board):
    # Resize the given board to be a nice rectangle with puyos of size multiple 8.
    height, _, _ = board.shape
    new_height = int(round(height/12/2/8)*12*2*8)
    new_width  = new_height/2
    resized = cv2.resize(board,(new_width,new_height))
    return resized

def getPuyoPos(board,pos):
    # Get the image of the puyo position on the 12 x 6 grid.
    height, width, _ = board.shape
    puyo_height = height/12
    puyo_width  = width/6
    x_range = ((pos[1]-1)*puyo_width,pos[1]*puyo_width)
    y_range = (height-(pos[0]-1)*puyo_height,height-pos[0]*puyo_height)
    puyo = board[y_range[1]:y_range[0],x_range[0]:x_range[1]]
    return puyo

def buildHOG(size):
    winSize = (size,size)
    blockSize = (size/4,size/4)
    blockStride = (size/8,size/8)
    cellSize = (size/8,size/8)
    nbins = 5
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    return hog

def trainSVM(board,hog,(pos_features,neg_features)):
    responses = []
    features  = []
    for pos in pos_features:
        puyo = getPuyoPos(board,pos)
        features.append(hog.compute(puyo))
        responses.append(1)
    for pos in neg_features:
        puyo = getPuyoPos(board,pos)
        features.append(hog.compute(puyo))
        responses.append(-1)
        
    features  = np.array(features,dtype=np.float32)
    responses = np.array(responses,dtype=np.int32)
    
    svm = cv2.ml.SVM_create()
    #svm.setKernel(cv2.ml.SVM_LINEAR)
    #svm.setType(cv2.ml.SVM_C_SVC)
    # svm.setC(2)
    # svm.setGamma(5)
    svm.trainAuto(features, cv2.ml.ROW_SAMPLE, responses)
    return svm

def trainSet((pos_features,neg_features)):
    pos_train = random.sample(pos_features,len(pos_features)/2)
    neg_train = []
    for neg_feature in neg_features:
        neg_train += random.sample(neg_feature,len(neg_feature)/2)
    return (pos_train, neg_train)

def countErrors(board, svm, hog, true_pos):
    errors = 0
    for row in range(1,13):
        for col in range(1,7):
            puyo    = getPuyoPos(board,(row,col))
            feature = np.transpose(np.array(hog.compute(puyo),dtype=np.float32))
            predict = float(svm.predict(feature)[1][0]) > 0
            if (row,col) in true_pos:
                if not predict:
                    errors += 1
            elif predict:
                    errors += 1
    return errors

def sweepFrame(frame, svm, hog, stepsize):
    height, width, _ = frame.shape
    size = height/12
    ycoords = np.arange(size/2, height-(size/2)+stepsize, stepsize)
    xcoords = np.arange(size/2,  width-(size/2)+stepsize, stepsize)
    xgrid, ygrid = np.meshgrid(xcoords,ycoords)
    zgrid = np.empty_like(xgrid,dtype=np.float32)
    for i, x in enumerate(xcoords):
        for j, y in enumerate(ycoords):
            subimage = frame[(y-(size/2)):(y+(size/2)), (x-(size/2)):(x+(size/2))]
            feature  = np.transpose(np.array(hog.compute(subimage),dtype=np.float32))
            predict  = np.float32(svm.predict(feature)[1][0])
            if predict > 0:
                zgrid[j,i] = predict
            else:
                zgrid[j,i] = np.nan
    zgrid = np.flip(zgrid,0)
    return (np.multiply(xgrid,zgrid), np.multiply(ygrid,zgrid))

def main():

    getImage()
    return
    
    board = getBoard()

    # Hardcoded from training image.
    blue    = [(4,2),(3,3),(2,4),(1,4),(1,5),(8,1),(7,1),(6,1),(4,4),(3,5),(3,6),(4,6),(9,5)]
    green   = [(1,1),(1,2),(2,3),(3,2),(4,1),(5,1),(5,2),(4,3),(6,3),(8,2),(6,6),(10,5),(11,6),(10,6),(9,6)]
    yellow  = [(2,1),(3,1),(2,2),(1,3),(6,2),(7,2),(5,4),(4,5),(5,6),(7,5),(8,5)]
    red     = [(3,4),(2,5),(2,6),(1,6),(6,4),(6,5),(5,5),(7,6),(8,6)]
    garbage = [(9,1),(5,3),(11,5)]
    empty   = [(12,1),(12,2),(12,3),(12,4),(12,5),(12,6),(11,1),(11,2),
               (11,3),(11,4),(10,1),(10,2),(10,3),(10,4),(9,2),(9,3),(9,4),(8,3),(8,4),(7,3),(7,4)]

    height, _, _ = board.shape
    hog = buildHOG(height/12)

    blue_svm = trainSVM(board, hog, trainSet((blue,[green,yellow,red,garbage,empty])))
    xgrid, ygrid = sweepFrame(board, blue_svm, hog, 32)
    plt.plot(xgrid,ygrid,'bo')

    green_svm = trainSVM(board, hog, trainSet((green,[blue,yellow,red,garbage,empty])))
    xgrid, ygrid = sweepFrame(board, green_svm, hog, 32)
    plt.plot(xgrid,ygrid,'go')

    red_svm = trainSVM(board, hog, trainSet((red,[green,yellow,blue,garbage,empty])))
    xgrid, ygrid = sweepFrame(board, red_svm, hog, 32)
    plt.plot(xgrid,ygrid,'ro')

    yellow_svm = trainSVM(board, hog, trainSet((yellow,[green,blue,red,garbage,empty])))
    xgrid, ygrid = sweepFrame(board, yellow_svm, hog, 32)
    plt.plot(xgrid,ygrid,'yo')

    #garbage_svm = trainSVM(board, hog, trainSet((garbage,[green,blue,red,yellow,empty])))
    #xgrid, ygrid = sweepFrame(board, garbage_svm, hog, 32)
    #plt.plot(xgrid,ygrid,'ko')
    
    plt.show()
    
    return

    accuracy = np.zeros((1,5))
    for i in range(0,100):
        blue_svm    = trainSVM(board, hog, trainSet((blue,[green,yellow,red,garbage,empty])))
        green_svm   = trainSVM(board, hog, trainSet((green,[blue,yellow,red,garbage,empty])))
        red_svm     = trainSVM(board, hog, trainSet((red,[blue,yellow,green,garbage,empty])))
        yellow_svm  = trainSVM(board, hog, trainSet((yellow,[blue,green,red,garbage,empty])))
        garbage_svm = trainSVM(board, hog, trainSet((garbage,[blue,yellow,red,green,empty])))

        blue_acc    = countErrors(board, blue_svm, hog, blue)
        green_acc   = countErrors(board, green_svm, hog, green)
        red_acc     = countErrors(board, red_svm, hog, red)
        yellow_acc  = countErrors(board, yellow_svm, hog, yellow)
        garbage_acc = countErrors(board, garbage_svm, hog, garbage)

        acc = np.array([blue_acc, green_acc, red_acc, yellow_acc, garbage_acc])
        accuracy += acc

    print accuracy

if __name__ == '__main__':
    main()

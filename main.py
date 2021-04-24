import heapq
import math
import os
import time
from multiprocessing import Process
import random

import cv2
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from numba import jit, int32, int64
# from numba.experimental import jitclass


# spec = [('r', int32), ('g', int32), ('b', int32), ('x', int32), ('y', int32)]

#@jitclass(spec)

# Obselete. Now using an array 5 wide. 0 is red, 1 is green, 2 is blue, 3 is x, 4 is y
class Pixel:

    # Data type to store basic info about a pixel in a video. Can add x,y, currently unnecessary.
    r = 0
    g = 0
    b = 0
    x = 0
    y = 0

    def __init__(self, r, g, b, x, y):
        self.r = r
        self.g = g
        self.b = b
        self.x = x
        self.y = y


def getFrames(fileName):
    # This function will take in the filename of a video. MP4 works for sure. If the file was 'movie.mp4', you'd give the param 'movie'
    # This function then will take the mp4, create folders for organization based on that name, and will store them in a folder with the same name.
    # We output a Frame#.tiff, as tiff is what we decided was best.
    getFramesStartTime = time.process_time()
    vidcap = cv2.VideoCapture(fileName + ".mp4")
    success, image = vidcap.read()
    count = 0
    folderAlteration = ""
    try:
        os.mkdir(fileName)
        # os.mkdir(fileName + '/Scatters')
        # print('Made Folder')
    except:
        folderAlteration = "1"
        print(
            'Folder Already Exists, writing instead to folder' + fileName + folderAlteration + ' to avoid overwrite issues. This file has atleasted start before.')
    while success:
        frameStartTime = time.process_time()
        cv2.imwrite(fileName + "/" + "frame" + str(count) + ".tiff", image)  # save frame as JPEG file
        success, image = vidcap.read()
        frameEndTime = time.process_time()
        print('Frame ' + str(count) + ' took ' + str((frameEndTime-frameStartTime)) + ' : ', success)
        count += 1
    getFramesEndTime = time.process_time()
    print('Gathered ' + str(count) + ' frame(s) in ' + str((getFramesEndTime - getFramesStartTime)) + ' seconds')


def frameGetAvgRGB(folderName, xLen, yLen, startingFrame):
    # i is the starting frame number. Default is 0 to
    i = startingFrame
    pixelCount = 0
    frame = cv2.imread(folderName + "/frame" + str(i) + ".tiff")
    # frame will be none if file doesn't exist. Check for that to prevent errors.
    if frame is None:
        print('Frame ' + str(i) + ' not found. Done Processing Frames')
        return False
    pixelList = []
    allR = 0
    allG = 0
    allB = 0
    for y in range(0, yLen):
        # print("Row " + str((x-1)) + " done.")
        for x in range(0, xLen):
            b, g, r = (frame[x, y])
            # print("RBG of pixel " + str(i) + " (" + str(r) + ", " + str(g) + ", " + str(b) + ")")
            allR += r
            allG += g
            allB += b
            pixelCount += 1
            pixelList.append(Pixel(r, g, b))

    print('AVG RGB of Frame ' + str(i) + ' = (' + str((allR / pixelCount)) + ',' + str((allG / pixelCount)) + ',' + str(
        (allB / pixelCount)) + ')')
    create3DScatterPlot(folderName, pixelList, i, 2)
    i += 1


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


def create3DScatterPlot(folderName, pixelList, frameNum, showOver):
    # Dump all duplicates.
    # showOver only plots the points if it occurs at least the specified amount
    # Create a 3d array 0-255, if we have one RGB value, set that value in the 3d array to 1, and don't add. Else, add and set it to 1.
    RBGArr = np.zeros((256, 256, 256))
    # newPixelList = []
    rL = []
    gL = []
    bL = []

    # print('Num Before = ' + str(len(pixelList)))
    i = 0
    for i in range(0, len(pixelList)):
        RBGArr[pixelList[i].r, pixelList[i].g, pixelList[i].b] += 1
        if (RBGArr[pixelList[i][0], pixelList[i][1], pixelList[i][2]] == showOver):
            # newPixelList.append(Pixel(pixelList[i].r, pixelList[i].g, pixelList[i].b))
            # RBGArr[pixelList[i].r, pixelList[i].g, pixelList[i].b] = 1
            rL.append(pixelList[i][0])
            gL.append(pixelList[i][1])
            bL.append(pixelList[i][2])

    print('Frame ' + str(frameNum) + ': ' + str(len(pixelList)) + '->' + str(len(rL)))

    fig = pyplot.figure()
    ax = Axes3D(fig)

    for i in range(0, len(rL)):
        rgbs = '#%02x%02x%02x' % (rL[i], gL[i], bL[i])
        ax.scatter(rL[i], gL[i], bL[i], c=rgbs)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')

    # pyplot.show()
    print('Saving Chart ' + str(frameNum) + '...')
    pyplot.savefig(folderName + '/Scatters/Frame' + str(frameNum) + '.png')
    print('Saved Chart for Frame ' + str(frameNum) + '.')
    pyplot.close(fig)


def createMP4FromPng(fileFormat, fps):
    image_folder = fileFormat
    video_name = fileFormat + 'Scatter.avi'
    print('Looking in ' + image_folder + '/Scatters/')
    images = [img for img in os.listdir(image_folder + '/Scatters/') if img.endswith(".png")]
    print('Found ' + str(len(images)))
    frame = cv2.imread((fileFormat + '/Scatters/Frame0.png'))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, fps, frameSize=(width, height))
    for x in range(0, len(images)):
        video.write(cv2.imread(fileFormat + '/Scatters/Frame' + str(x) + '.png'))

    cv2.destroyAllWindows()
    video.release()


def insertIntoDB(pixel):
    import MySQLdb as mysql

    mydb = mysql.connector.connect(
        host="localhost",
        user="yourusername",
        password="yourpassword"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SHOW DATABASES")
    for x in mycursor:
        print(x)


def getFramesWithPath(fileName, absolutePath):
    # This function will take in the filename of a video. MP4 works for sure. If the file was 'movie.mp4', you'd give the param 'movie'
    # This function will also take in the absolute path to place tiff files. To put a file movie.mp4 into D:/Movies, you'd call getFramesWithPath('movie','D:/Movies')
    # This function then will take the mp4, create folders for organization based on that name, and will store them in a folder with the same name.
    # We output a Frame#.tiff, as tiff is what we decided was best.

    getFramesStartTime = time.process_time()
    vidcap = cv2.VideoCapture(fileName + ".mp4")
    success, image = vidcap.read()
    count = 0
    folderAlteration = ""
    try:
        os.mkdir(absolutePath + '/' + fileName)
        # os.mkdir(absolutePath + '/' + fileName + '/Scatters')
        # print('Made Folder')
    except:
        print('File has began to be processed before. Please Delete Folder and retry.')

    while success:
        frameStartTime = time.process_time()
        cv2.imwrite(absolutePath + '/' + fileName + "/" + "frame" + str(count) + ".tiff",
                    image)  # save frame as JPEG file
        success, image = vidcap.read()
        frameEndTime = time.process_time()
        # print('Frame ' + str(count) + ' took ' + str((frameEndTime-frameStartTime)) + ' : ', success)
        count += 1
    getFramesEndTime = time.process_time()
    print('Gathered ' + str(count) + ' frame(s) in ' + str((getFramesEndTime - getFramesStartTime)) + ' seconds')


# @jit('i8[:,:](i4,i4,types.unicode_type,i4)', nopython=False)
def loadPixelArray(width, length, folder, frameNum):
    # import cupy as np

    # function returns an array of Pixel objects for a frame.
    # i is the starting frame number. Default is 0 to
    timeStart = time.time()
    pixelCount = 0
    frame = cv2.imread(folder + "/frame" + str(frameNum) + ".tiff")
    # frame will be none if file doesn't exist. Check for that to prevent errors.
    if frame is None:
        print('Frame ' + str(frameNum) + ' not found at ' + folder + "/frame" + str(
            frameNum) + ".tiff" + '. Done Processing Frames')
        return False
    pixelList = np.empty(shape=(length*width,5), dtype=int)
    r = 0
    g = 0
    b = 0

    for y in range(0, width):
        # print("Row " + str((x-1)) + " done.")
        for x in range(0, length):
            b, g, r = (frame[x, y])
            pixelList[pixelCount][0] = r
            pixelList[pixelCount][1] = g
            pixelList[pixelCount][2] = b
            pixelList[pixelCount][3] = x
            pixelList[pixelCount][4] = y

            pixelCount += 1

        # print('Y:' + str(y))
    print('loadPixelArray took ' + str(time.time() - timeStart) + ' seconds.')
    return np.array(pixelList)

@jit()
def returnTopTen(width, height, fileName, frameNum):
    # import cupy as np
    pixels = loadPixelArray(width, height, fileName, frameNum)
    topTen = np.empty((10, 5), dtype='i8')
    # create array of zeroes

    tracker = np.zeros((256 * 256 * 256), dtype='i8')
    # timeStart = time.process_time()
    # print('Starting top ten process..')
    # tracker int array will be equal to the occurrences of the [r][g][b] value.
    #if not pixels:
    #    return [Pixel(-1, -1, -1, -1, -1)]
    for i in range(0, len(pixels)):
        tracker[((pixels[i][0] * 65536) + (pixels[i][1] * 256) + pixels[i][2])] += 1  # add one to count the occurrence of a given RBG value
    ind = tracker.argsort()[::-1][:10]
    for i in range(0, len(ind)):
        r = math.floor(ind[i] / 65536)  # round down on both, will always be partial if not like [255][0][0] or [156][0][0]
        g = math.floor((ind[i] / 256) % 256)
        b = ind[i] % 256
        np.append(topTen, [r, g, b, int(tracker[ind[i]]), 0])

        print(str(i + 1) + ': (' + str(r) + ', ' + str(g) + ', ' + str(b) + ') occurs ' + str(int(tracker[ind[i]])) + ' times.')
    # print('Top Ten Calculation took ' + str(time.process_time() - timeStart) + ' seconds.')
    getNKMeans(pixels, 10, frameNum, 'madMax', 800, 1920)
    return topTen
    # Return topTen

    # print('ind: ' + str(ind))


def createPalleteImage(colors, width, height, folder, frameNum):
    from colorsys import hsv_to_rgb
    from PIL import Image
    barW = math.floor((width / len(colors)))
    imageA = np.zeros([height, width, 3], dtype=np.uint8)
    holder = 0
    for i in range(0, len(colors)):
        imageA[0:height, holder:(holder + barW)] = [colors[i, 0], colors[i, 1], colors[i, 2]]
        holder += barW

    img = Image.fromarray(imageA, 'RGB', )
    img.save(folder + '/palettes/' + str(frameNum)+'.png')
    img.show()


def makeCharts(frameCount, fileName):
    while True:
        threads = []
        threadCount = 10
        errors = 0
        for x in range(0, threadCount):
            print('Starting Thread ' + str(x))
            thread = Process(target=frameGetAvgRGB, args=(fileName, 1080, 1920, frameCount + x))
            threads.append(thread)
            threads[x].start()
        for x in range(0, threadCount):
            threads[x].join()
        print('Completed Frames ' + str(frameCount) + '-' + str(frameCount + 9))
        frameCount += 10


def threadedGetTopTen(width, height, fileName, frameNum):
    threadCount = 1
    while True:
        timeStart = time.time()
        threads = []
        for i in range(0, threadCount):
            thread = Process(target=returnTopTen, args=(width, height, fileName, frameNum + i))
            threads.append(thread)
            # print('Starting ' + str(frameNum + i))
            threads[i].start()
            # print('Done starting ' + str(i))
        for i in range(0, threadCount):
            threads[i].join()
        print(str(threadCount) + ' frames took ' + str(round(time.time() - timeStart, 4)) + ' seconds. AVG:' + str(
            round((time.time() - timeStart) / threadCount, 4)) + ' seconds/frame.')
        frameNum += threadCount

@jit()
def getNKMeans(pixels, clusterNum, frameNum, folder, width, height):
    clusters = np.zeros([clusterNum, 3], dtype=np.uint8)
    frame = cv2.imread(folder + "/frame" + str(frameNum) + ".tiff")
    print('Looking for "' + folder + "/frame" + str(frameNum) + ".tiff\"")
    # frame will be none if file doesn't exist. Check for that to prevent errors.
    if frame is None:
        print('Frame ' + str(frameNum) + ' not found at ' + folder + "/frame" + str(
            frameNum) + ".tiff" + '. Done Processing Frames')
        return False
    # Set the initial N clusters
    for i in range(0, clusterNum):
        x = random.randrange(0, width)
        y = random.randrange(0, height)
        b, g, r = (frame[x, y])
        clusters[i, 0] = r
        clusters[i, 1] = g
        clusters[i, 2] = b
        # print('Frame: "' + str(frameNum[x,y]))
        print('Pixel chosen at (' + str(x) + ', ' + str(y) + ') Color: (' + str(r) + ', ' + str(g) + ', ' + str(b) + ')')

    # Variable setup, same structure, just 64 bits to store massive numbers just in case.
    clusterAvgs = np.zeros([clusterNum, 3], dtype=np.int64)
    clusterCount = np.zeros([clusterNum], dtype=np.int32)
    counter = 1
    changed = True
    while(changed):
        changed = False
        print('K Means iteration ' + str(counter))
        counter += 1
        for pixel in pixels:
            # get distance to each point.
            d = 100000
            clusterMatch = 0
            for i in range(0, clusterNum):
                tR = (clusters[i, 0] - pixel[0]) ** 2
                tG = (clusters[i, 1] - pixel[1]) ** 2
                tB = (clusters[i, 2] - pixel[2]) ** 2
                dT = math.sqrt(tR + tG + tB)
                if dT < d:
                    d = dT
                    clusterMatch = i
            # print('Closest to cluster ' + str(clusterMatch))
            # Now we know which cluster the pixel is closest to, so add values to the total for that cluster
            clusterAvgs[clusterMatch, 0] += pixel[0]
            clusterAvgs[clusterMatch, 1] += pixel[1]
            clusterAvgs[clusterMatch, 2] += pixel[2]
            # Add one to counter to get a running total.
            clusterCount[clusterMatch] += 1
        for i in range(0, clusterNum):
            print('Cluster ' + str(i) + ' (' + str(round(clusterAvgs[i, 0]/clusterCount[i], 2)) + ', ' + str(round(clusterAvgs[i, 1]/clusterCount[i], 2)) + ', ' + str(round(clusterAvgs[i, 2]/clusterCount[i], 2)) + ')')
            # reset Clusters for repeating to new values,
            if round(clusters[i, 0]) != round(clusterAvgs[i, 0] / clusterCount[i]):
                changed = True
                clusters[i, 0] = round(clusterAvgs[i, 0] / clusterCount[i])
            if round(clusters[i, 1]) != round(clusterAvgs[i, 1] / clusterCount[i]):
                changed = True
                clusters[i, 1] = round(clusterAvgs[i, 1] / clusterCount[i])
            if round(clusters[i, 2]) != round(clusterAvgs[i, 2] / clusterCount[i]):
                changed = True
                clusters[i, 2] = round(clusterAvgs[i, 2] / clusterCount[i])


    # reorder pallette
    clusters = np.array(sorted(clusters, key=lambda row: sum(row)))
    for j in range(clusterNum):
        print('Sorted Value: ' + str(clusters[j, 0] + clusters[j, 1] + clusters[j, 2]))

    createPalleteImage(clusters, 1920, 280, folder, frameNum)






if __name__ == '__main__':
    # main execution thread
    # import colorDatabase


    # getFrames() takes in the name of the file, without the extension mp4, creates a folder, and then fills the folder with all frames.

    # getFrames('1917')
    # getFrames('AnnaKarenina')
    # getFrames('Arrival')
    # getFrames('BladeRunner2049')
    # getFrames('DarkestHour')
    # getFrames('DjangoUnchained')
    # getFrames('Gravity')
    # getFrames('Hugo')
    # getFrames('Inception')
    # getFrames('InsideLlewynDavis')
    # getFrames('Joker')
    # getFrames('LifeofPi')
    # getFrames('Lincoln')
    # getFrames('MadMaxFuryRoad')
    # getFrames('Sicario')
    # getFrames('Tenet')
    # getFrames('TheHatefulEight')
    # getFrames('TheShapeofWater')

    # Run multiple threads

    # Example of a Pixel Array filled for a given array, function can be used as a parameter in other functions, see next example.
    # a = loadPixelArray(1920, 800, 'MadMaxFuryRoad', 600)
    # print(' b: ' + str(a[50000].b))

    # Example of a Pixel Array being filled by loadPixelArray, but the returned pixel array is then sent to the returnTopTen function.
    # getFrames('tonyDies')
    # returnTopTen(loadPixelArray(1920, 800, 'MadMaxFuryRoad', 600))

    # calling a function that runs returnTopTen on multiple threads, because parallelism is much faster.


    threadedGetTopTen(1920, 800, 'madMax', 24861)




    #getFrames('madMax')

    # Line below will open (threadCount) number of threads to process frames into the graph. Leave as false for now, will move to a function later.

    # looks in specified folder for 'folder'/Scatters/Frame#.png, will turn all of them into a avi file. 2nd param is fps.
    # createMP4FromPng('frozen', 23.98)





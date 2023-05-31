import heapq
import math
import os
import time
from multiprocessing import Process
import random
import cv2
import numpy as np

#from insert import *
from randomPalettes import *


allKMeans = []
allTopTen = []
allPicked = []
margin = 20

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
        print('Frame ' + str(count) + ' took ' + str((frameEndTime - frameStartTime)) + ' : ', success)
        count += 1
    getFramesEndTime = time.process_time()
    print('Gathered ' + str(count) + ' frame(s) in ' + str((getFramesEndTime - getFramesStartTime)) + ' seconds')


def getFramesInterval(fileName, fps, movieLength, totalFrames, padding, startingTime):
    framesSkipped = round((fps * float(movieLength)) / totalFrames)
    vidcap = cv2.VideoCapture(fileName + ".mp4")
    success, image = vidcap.read()
    count = 0
    done = False
    getFramesStartTime = time.time()
    groups = 0
    folderAlteration = ""
    try:
        os.mkdir(fileName)
        # os.mkdir(fileName + '/Scatters')
        # print('Made Folder')
    except:
        folderAlteration = "1"
        print('Folder Already Exists, please deleted and restart.')
        return 1
    # initial skip to starting frames of actual content

    initSkip = round(startingTime * fps)
    for i in range(0, initSkip):
        count += 1
        success, image = vidcap.read()
        if not success:
            done = True
            break

    while not done:
        # Grab frames
        for i in range(-padding, padding + 1):
            cv2.imwrite(fileName + "/" + "frame" + str(count) + ".tiff", image)  # save frame as TIFF file
            count += 1
            success, image = vidcap.read()
            if not success:
                done = True
                break

        for i in range(0, framesSkipped - padding):
            success, image = vidcap.read()
            count += 1
            if not success:
                done = True;
                break
        groups += 1
        if groups >= totalFrames:
            done = True

    getFramesEndTime = time.time()
    print('Gathered ' + str(count) + ' frame(s) in ' + str((getFramesEndTime - getFramesStartTime)) + ' seconds')


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
        os.mkdir(absolutePath + '/' + fileName + '/palettes')

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


def loadPixelArray(width, height, folder, fileName):
    # import cupy as np
    # function returns an array of Pixel objects for a frame.
    # i is the starting frame number. Default is 0 to
    timeStart = time.time()
    pixelCount = 0
    frame = cv2.imread(folder + "/" + str(fileName))
    # frame will be none if file doesn't exist. Check for that to prevent errors.
    if frame is None:
        print('Frame ' + str(fileName) + ' not found at ' + folder + '/' + fileName + '. Done Processing Frames')
        return False
    pixelList = np.empty(shape=(height * width, 5), dtype=int)
    r = 0
    g = 0
    b = 0

    for y in range(0, height):
        # print("Row " + str((x-1)) + " done.")
        for x in range(0, width):
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


def returnTopTen(width, height, fileName, frameNum):
    # import cupy as np
    pixels = loadPixelArray(width, height, fileName, frameNum)
    topTen = np.empty([10, 5], dtype=np.uint8)
    # create array of zeroes

    tracker = np.zeros((256, 256, 256), dtype=np.uint8)
    # tracker int array will be equal to the occurrences of the [r][g][b] value.
    for i in range(0, len(pixels)):
        tracker[((pixels[i][0] * 65536) + (pixels[i][1] * 256) + pixels[i][
            2])] += 1  # add one to count the occurrence of a given RBG value
    ind = tracker.argsort()[::-1][:10]
    for i in range(0, len(ind)):
        r = math.floor(
            ind[i] / 65536)  # round down on both, will always be partial if not like [255][0][0] or [156][0][0]
        g = math.floor((ind[i] / 256) % 256)
        b = ind[i] % 256
        np.append(topTen, [r, g, b, int(tracker[ind[i]]), 0])

        print(str(i + 1) + ': (' + str(r) + ', ' + str(g) + ', ' + str(b) + ') occurs ' + str(
            int(tracker[ind[i]])) + ' times.')
    # print('Top Ten Calculation took ' + str(time.process_time() - timeStart) + ' seconds.')
    getNKMeans(pixels, 10, str(frameNum), fileName, 800, 1920)
    return topTen


def createPalleteImage(colors, width, height, folder, fileName):
    from colorsys import hsv_to_rgb
    from PIL import Image
    try:
        barW = math.floor((width-margin*9) / len(colors))
        imageA = np.zeros([height, width, 3], dtype=np.uint8)
        
        holder = 0
        for i in range(0, len(colors)):
            imageA[0:height, holder:(holder + barW)] = [colors[i, 0], colors[i, 1], colors[i, 2]]
            holder += barW
            # Eva: add white merge bar between each color
            if(i<len(colors)-1):
                cv2.rectangle(imageA, (int(holder), 0), (int(holder + margin), height),(255, 255, 255), -1)
                holder+= margin
        img = Image.fromarray(imageA, 'RGB')
        img.save(folder + '/palettes/' + fileName + '.png')
    except Exception as e:
        print("Error: ", e)
    # img.show()


def threadedGetTopTen(width, height, fileName, frameNum):
    threadCount = 5
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


def getNKMeans(pixels, clusterNum, folder, fileName, width, height):
    clusters = np.zeros([clusterNum, 3], dtype=np.uint8)
    frame = cv2.imread(folder + "/" + str(fileName))
    print('Looking for "' + folder + "/" + str(fileName) + "\"")
    # frame will be none if file doesn't exist. Check for that to prevent errors.
    if frame is None:
        print('Frame ' + str(fileName) + ' not found at ' + folder + "/frame" + str(
            fileName) + '. Done Processing Frames')
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
        print(
            'Pixel chosen at (' + str(x) + ', ' + str(y) + ') Color: (' + str(r) + ', ' + str(g) + ', ' + str(b) + ')')

    # Variable setup, same structure, just 64 bits to store massive numbers just in case.
    # clusterAvgs = np.zeros([clusterNum, 3], dtype=np.int64)
    # clusterCount = np.zeros([clusterNum], dtype=np.int32)
    counter = 0
    changed = True
    while changed:
        clusterAvgs = np.zeros([clusterNum, 3], dtype=np.int64)
        clusterCount = np.zeros([clusterNum], dtype=np.int32)
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
            # reset Clusters for repeating to new values,
            if not (clusterCount[i] > 0) or clusterCount[i] is None:
                clusterCount[i] = 1
            r = round((clusterAvgs[i, 0] / clusterCount[i]))
            g = round(clusterAvgs[i, 1] / clusterCount[i])
            b = round(clusterAvgs[i, 2] / clusterCount[i])
            if round(clusters[i, 0]) != r:
                changed = True
                clusters[i, 0] = clusterAvgs[i, 0] / clusterCount[i]
            if round(clusters[i, 1]) != g:
                changed = True
                clusters[i, 1] = clusterAvgs[i, 1] / clusterCount[i]
            if round(clusters[i, 2]) != b:
                changed = True
                clusters[i, 2] = clusterAvgs[i, 2] / clusterCount[i]

        # Comment out next line if you don't need a palette of each iteration.
        clusters = np.array(sorted(clusters, key=lambda row: max(row)))  # sort by brightest
        clusters = clusters[::-1]
        # createPalleteImage(clusters, 1920, 280, folder, (str(fileName) + 'Iteration' + str(counter)))
        print('Iteration Done')
        if counter == 15:
            changed = False


    # reorder pallette
    clusters = np.array(sorted(clusters, key=lambda row: max(row)))
    clusters = clusters[::-1]
    for j in range(clusterNum):
        print('Sorted Value: ' + str(int(clusters[j, 0])+ int(clusters[j, 1]) + int(clusters[j, 2])))
    return clusters


def readMovieCSV(fileName):
    import csv
    threads = []
    file = open(fileName)
    csvreader = csv.reader(file)
    header = next(csvreader)
    print(header)
    rows = []
    count = 0
    for row in csvreader:
        rows.append(row)
    for row in rows:
        print('Processing Movie ' + str(row[0]))
        length = row[4].split(':')
        # seconds of actual footage
        showTime = (3600 * int(length[0])) + (60 * int(length[1])) + int(length[2])
        print('H: ' + str(length[0]) + ' M: ' + str(length[1]) + ' S: ' + str(length[2]))
        print('Showtime in Seconds: ' + str(showTime))
        start = row[2].split(':')
        startTime = (60 * int(start[1])) + int(start[2])
        fName = (row[1].replace(".mp4", ""))
        fps = float(row[5])
        thread = Process(target=getFramesInterval, args=(fName, fps, showTime, 20, 3, startTime))
        threads.append(thread)
        # print('Starting ' + str(frameNum + i))
        threads[count].start()
        # getFramesInterval(fName, fps, showTime, 20, 3, startTime)
        count += 1
    file.close()
    for i in range(0, count):
        threads[i].join()


def getFileNames(folderName):
    from pathlib import Path

    # iterate over files in
    # that directory
    files = sorted(Path(folderName).glob('*.tiff'), key=lambda x: int(os.path.splitext(x)[0].split('ame')[1]))

    return files


def processFrames(folderName,csvfile, count):
    files = list(getFileNames(folderName))

    # print('File 0: ' + str(files[0]))
    # for filePath in files:
    #     filePath = str(filePath).split('\\')
    # processSingleFrame(800,1920, filePath[0], filePath[1])
    threads = []
    for i in range(0, len(files)):
        filePath = str(files[i]).split('/')
        thread = Process(target=processSingleFrame, args=(1036, 1920, filePath[0], filePath[1], csvfile, count))

        threads.append(thread)
        threads[i].start()
    for i in range(0, len(files)):
        threads[i].join()


def processSingleFrame(height, width, folder, fileName, csvfile, count):
    # import cupy as np
    try:
        pixels = loadPixelArray(height, width, folder, fileName)
        topTen = np.empty([10, 3], dtype=np.uint8)
        #create array of zeroes
        tracker = np.zeros((256,256,256), dtype=int)
        # tracker int array will be equal to the occurrences of the [r][g][b] value.
##        for i in range(0, len(pixels)):
##            tracker[pixels[i][0]][pixels[i][1]][pixels[i][2]] += 1 # add one to count the occurrence of a given RBG value
##
##        for num in range(0, 10): # get 10 highest values
##            maxR = 0
##            maxG = 0
##            maxB = 0
##            highestVal = 0
##            for i in range(255):
##                for j in range(255):
##                    for k in range(255):
##                        if tracker[i][j][k] > highestVal:
##                            highestVal = tracker[i][j][k]
##                            maxR = i
##                            maxG = j
##                            maxB = k
##            print('[' + str(num) + ': ' + str(maxR)+',' + str(maxG) + ',' + str(maxB) + '] ' + str(highestVal) + ' times.')
##            tracker[maxR][maxG][maxB] = 0 # set to -1 so it wont ever be picked again, avoid duplicates
##            topTen[num][0] = maxR
##            topTen[num][1] = maxG
##            topTen[num][2] = maxB
##        topTen = np.array(sorted(topTen, key=lambda row: max(row)))
##        topTen = topTen[::-1]
##
##        createPalleteImage(topTen, 1920, 280, folder, fileName + "-TOPTEN")
        # reading random pick .csv file from randomPalettes.py
        # allPicked = randomP(csvfile)
        # randomPalette = allPicked[count]
        # #print("random: ", count, fileName, randomPalette)
        # createPalleteImage(randomPalette, 1920, 280, folder, fileName + "-PICKED")

        KMeansClusters = getNKMeans(pixels, 10, folder, fileName, height, width)

        # Get the Chosen Palette.

        # Create Palette Images
        createPalleteImage(KMeansClusters, 1920, 280, folder, fileName + "-KMEANS(15)")
    except Exception as e:
        print("single error: ", e)
    return topTen


def overallKmeans(pixels, clusterNum):
    clusters = np.zeros([clusterNum, 3], dtype=np.uint8)
    for i in range(0, clusterNum):
        clusters[i, 0] = pixels[i, 0]
        clusters[i, 1] = pixels[i, 1]
        clusters[i, 2] = pixels[i, 2]

    counter = 0
    changed = True
    while changed:
        clusterAvgs = np.zeros([clusterNum, 3], dtype=np.int64)
        clusterCount = np.zeros([clusterNum], dtype=np.int32)
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
                print("r:", tR, "g:", tG, "b:", tB)
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
            # reset Clusters for repeating to new values,
            if not (clusterCount[i] > 0) or clusterCount[i] is None:
                clusterCount[i] = 1
            r = round((clusterAvgs[i, 0] / clusterCount[i]))
            g = round(clusterAvgs[i, 1] / clusterCount[i])
            b = round(clusterAvgs[i, 2] / clusterCount[i])
            if round(clusters[i, 0]) != r:
                changed = True
                clusters[i, 0] = clusterAvgs[i, 0] / clusterCount[i]
            if round(clusters[i, 1]) != g:
                changed = True
                clusters[i, 1] = clusterAvgs[i, 1] / clusterCount[i]
            if round(clusters[i, 2]) != b:
                changed = True
                clusters[i, 2] = clusterAvgs[i, 2] / clusterCount[i]

        # Comment out next line if you don't need a palette of each iteration.
        clusters = np.array(sorted(clusters, key=lambda row: max(row)))  # sort by brightest
        clusters = clusters[::-1]
        # createPalleteImage(clusters, 1920, 280, folder, (str(fileName) + 'Iteration' + str(counter)))
        print('Iteration Done')
        if counter == 15:
            changed = False

    # reorder pallette
    clusters = np.array(sorted(clusters, key=lambda row: sum(row)))
    clusters = clusters[::-1]
    for j in range(clusterNum):
        print('Sorted Value: ' + str(int(clusters[j, 0]) + int(clusters[j, 1]) + int(clusters[j, 2])))
    return clusters

if __name__ == '__main__':
    # main execution thread
    # getFrames() takes in the name of the file, without the extension mp4, creates a folder, and then fills the folder with all frames.
    # threadedGetTopTen(1920, 800, 'Inception', 2077)

    # Example of a Pixel Array filled for a given array, function can be used as a parameter in other functions, see next example.
    # a = loadPixelArray(1920, 800, 'MadMaxFuryRoad', 600)
    # Example of a Pixel Array being filled by loadPixelArray, but the returned pixel array is then sent to the returnTopTen function.
    # getFrames('tonyDies')
    # returnTopTen(loadPixelArray(1920, 800, 'MadMaxFuryRoad', 600))

    #getFramesInterval("AnnaKarenina", 23.98, 9059, 20, 3, 123)

    # ***** Example of running all movies in a csv. *****
    # readMovieCSV('MovieData.csv')

    processSingleFrame(1036, 1920, "Joker", "frame138601.tiff", "", 0)
    
    #processFrames('TheShapeofWater', 'hateful.csv', 4)
    # allPicked = randomP("gravity.csv")


    # for i in range(15):
    #     array[i] = array[i] = np.array(sorted(array[i], key=lambda x: sum(x), reverse=True))
    #     createPalleteImage(array[i], 1920, 280, "picked", str(id[i]) + "-avgPICKED")

    # array[13] =np.array(sorted(array[13], key=lambda x: sum(x), reverse=True))

from turtle import width
import cv2
import numpy as np
import ctypes


##################### Setting Image Path Or Camera ###############################
webCamFeed = False
dirPath = "Documents/"
imgName = "2.jpg"
cap = cv2.VideoCapture(1)
cap.set(10,160)

##################################################################################




################################### Functions ####################################



# Getting Screen Resolution
user32 = ctypes.windll.user32
screenWidth, screenHeight = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)




def resizeFrame(img) :
    widthImg, heightImg = img.shape[1], img.shape[0]

    scale = 1 
    if heightImg>widthImg :
        scale = screenHeight/heightImg
    else :
        scale = screenWidth/widthImg

        
    widthImg *= scale
    heightImg *= scale

    dimensions = (int(widthImg/2), int(heightImg/2))
    return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)




def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew





def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area





def drawRectangle(img,biggest,thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img





def nothing(x):
    pass




def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 380, 80)
    cv2.createTrackbar("Threshold1", "Trackbars", 200,255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)




def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1,Threshold2
    return src


########################################################################

initializeTrackbars()
count=0




while True:

    if webCamFeed:success, img = cap.read()
    else:img = cv2.imread(dirPath + imgName)
    heightImg = img.shape[0]
    widthImg = img.shape[1]
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    thres=valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1]) # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

    ## FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS


    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        biggest=reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        imgBigContour = drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # cv2.imshow("1st warp Coloured", imgWarpColored)

        #REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))
        # imgWarpColored = resizeFrame( imgWarpColored, 0.20 )
        
        # cv2.imshow("1st warp Coloured", imgWarpColored)

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)

        # Image Array for Display
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre, imgWarpGray])

    else:
        imgBlank = resizeFrame(imgBlank)
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank, imgBlank])


    cv2.imshow("Original",resizeFrame(img))
    # cv2.imshow("Original",img)
    cv2.imshow("Output Coloured",resizeFrame(imageArray[1][1]))
    cv2.imshow("Output Binary",resizeFrame(imageArray[1][3]))
    cv2.imshow("Output Grayscale",resizeFrame(imageArray[1][4]))
    cv2.imshow("Original Image Gray",resizeFrame(imageArray[0][1]))
    cv2.imshow("Sevaral Image Contours",resizeFrame(imageArray[0][3]))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # SAVE IMAGE WHEN 's' key is pressed
    elif (cv2.waitKey(1) & 0xFF == ord('s')):
        cv2.imwrite("myImage"+str(count)+".jpg",imageArray[1][1])
        cv2.waitKey(300)
        count += 1
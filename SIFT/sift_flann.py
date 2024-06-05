import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(0)

print("video width=", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("video height=", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

img1_path = sys.argv[1]
img1 = cv2.imread(img1_path, 0)

sift = cv2.SIFT_create()
#sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    ret, frame = cap.read()

    if ret != True:
        print("ret is NOT True")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(frame, None)

    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    for i, (m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1,0]

    draw_params = dict(matchColor = (0, 255, 0),
            singlePointColor = (255, 0, 0),
            matchesMask = matchesMask,
            flags = 0)
    matched_img = cv2.drawMatchesKnn(img1, kp1, frame, kp2, matches, None, **draw_params)
    cv2.imshow('matched_img', matched_img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break





import os, sys
from matplotlib import pyplot as plt
from skimage import feature, color, transform, io
import cv2
import numpy as np

# why cv2 is giving me fucking read error?
# im = cv2.imread('a.png')
im = io.imread('a.png')


def add_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print('add_point({}, {})'.format(x, y), file = sys.stderr)

cv2.setMouseCallback('input', add_point)

points = [
    (257, 262),
    (383, 226),
    (265, 370),
    (397, 363),
]
cv2.line(im, points[0], points[1], (255, 0, 255))
cv2.line(im, points[2], points[3], (255, 0, 255))
cv2.line(im, points[0], points[2], (255, 255, 0))
cv2.line(im, points[1], points[3], (255, 255, 0))
cv2.imshow('input', im)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def scale(A, s):
    ret = A.copy()
    ret[0,0] *= s
    ret[1,1] *= s
    return ret

def translate(A, x, y):
    ret = A.copy()
    ret[0,2] += x
    ret[1,2] += y
    print('H =', ret)
    return ret


v1 = np.array(line_intersection([points[0], points[1]], [points[2], points[3]]) + (1,))
v2 = np.array(line_intersection([points[0], points[2]], [points[1], points[3]]) + (1,))
f = np.sqrt(1 - np.dot(v1, v2))
K = np.diag([f, f, 1])
r1 = np.dot(np.diag(np.diag(K)**(-1)), v1)
r1 /= np.linalg.norm(r1)
r2 = np.dot(np.diag(np.diag(K)**(-1)), v2)
r2 /= np.linalg.norm(r2)
r3 = np.cross(r1, r2)
print('<r1 r2>', np.dot(r1, r2))
R = np.column_stack([r1, r2, r3])
print('det(R) =', np.linalg.det(R))
H = np.dot(K, np.dot(np.linalg.inv(R), np.linalg.inv(K)))
# H = np.dot(K, np.dot(R, np.linalg.inv(K)))
# H[2,0] = H[2,1] = 0
print('(H) =', H)
def showRectified(H, idx):
    imOut = cv2.warpPerspective(im, H, (1024,768))
    cv2.imshow('output%d'%idx, imOut)

# showRectified(translate(scale(H, 0.5), -1200, 100), 0)
showRectified(translate(scale(H, 0.5), -1200, -100), 1)
# showRectified(translate(scale(H, 0.5), -1200, -200), 2)
cv2.waitKey(0)

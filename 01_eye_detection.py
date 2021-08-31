import cv2
import dlib
import numpy as np
import math

cap = cv2.VideoCapture(0)

detectors = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_ITALIC

# get mid-points of eye
def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

# get blinking
def blinking(eye_pt, facial_landmarks):
    left_side_eye = (facial_landmarks.part(eye_pt[0]).x, facial_landmarks.part(eye_pt[0]).y)
    right_side_eye = (facial_landmarks.part(eye_pt[3]).x, facial_landmarks.part(eye_pt[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_pt[1]), facial_landmarks.part(eye_pt[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_pt[5]), facial_landmarks.part(eye_pt[4]))
    hor_line = cv2.line(frame, left_side_eye, right_side_eye, (255,0,0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (255,0,0), 2)
    hor_line_len = math.hypot((left_side_eye[0] - right_side_eye[0]), (left_side_eye[1] - right_side_eye[1]))
    ver_line_len = math.hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_len / ver_line_len
    return ratio


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detectors(gray)
    for face in faces:
        # x, y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x,y), (x1, y1), (255,0,0), 1)
        # to detect the landmarks on face
        landmarks = predictor(gray, face)
        left_eye_ratio = blinking([36,37,38,39,40,41], landmarks)
        right_eye_ratio = blinking([42,43,44,45,46,47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) /2
        if blinking_ratio > 5.7:
            cv2.putText(frame, "Blinking", (50, 150), font, 7, (255,0,0))
        # x = landmarks.part(36).x
        # y = landmarks.part(36).y
        # cv2.circle(frame, (x,y), 3, (255,0,0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.realease()
cv2.destroyAllWindows()
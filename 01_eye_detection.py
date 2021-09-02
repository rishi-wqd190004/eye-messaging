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
    #hor_line = cv2.line(frame, left_side_eye, right_side_eye, (255,0,0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (255,0,0), 2)
    hor_line_len = math.hypot((left_side_eye[0] - right_side_eye[0]), (left_side_eye[1] - right_side_eye[1]))
    ver_line_len = math.hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_len / ver_line_len
    return ratio

# gaze detection region
def find_gaze_ratio(eye_pt, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_pt[0]).x, facial_landmarks.part(eye_pt[0]).y), 
                                    (facial_landmarks.part(eye_pt[1]).x, facial_landmarks.part(eye_pt[1]).y),
                                    (facial_landmarks.part(eye_pt[2]).x, facial_landmarks.part(eye_pt[2]).y),
                                    (facial_landmarks.part(eye_pt[3]).x, facial_landmarks.part(eye_pt[3]).y),
                                    (facial_landmarks.part(eye_pt[4]).x, facial_landmarks.part(eye_pt[4]).y),
                                    (facial_landmarks.part(eye_pt[5]).x, facial_landmarks.part(eye_pt[5]).y)], np.int32)
    #cv2.polylines(frame, [left_eye_region], True, (255,255,255), 2)

    # mask
    ht, wt, _ = frame.shape
    mask = np.zeros((ht, wt), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    #eye = frame[min_y: max_y, min_x: max_x]
    gray_eye = eye[min_y: max_y, min_x: max_x]
    #gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    _, th_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    threshold_eye_1 = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    threshold_eye_2 = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)

    height, width = threshold_eye_1.shape
    left_side_th = threshold_eye_1[0: height, 0: int(width/2)]
    left_side_white = cv2.countNonZero(left_side_th)


    right_side_th = threshold_eye_1[0: height, int(width/2): width]
    right_side_white = cv2.countNonZero(right_side_th)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white

    # cv2.putText(frame, str(left_side_white), (50,300), font, 2, (0,0,255), 3)
    # cv2.putText(frame, str(right_side_white), (50,350), font, 2, (0,0,255), 3)
    # gaze_ratio = left_side_white / right_side_white
    # cv2.putText(frame, str(gaze_ratio), (50,200), font, 2, (0,255,255), 3)
    return gaze_ratio

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_frame = np.zeros((500,500,3), np.uint8)
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

        # gaze detection
        gaze_ratio_left_eye = find_gaze_ratio([36,37,38,39,40,41], landmarks)
        gaze_ratio_right_eye = find_gaze_ratio([42,43,44,45,46,47], landmarks)

        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
        if gaze_ratio <= 1:
            cv2.putText(frame, "RIGHT", (50,100), font, 2, (0,255,255), 3)
            new_frame[:] = (0,0,255)
        elif 1 < gaze_ratio < 1.7:
            cv2.putText(frame, "CENTER", (50,100), font, 2, (0,255,255), 3)
        else:
            new_frame[:] = (255,0,0)
            cv2.putText(frame, "LEFT", (50,100), font, 2, (0,255,255), 3)
        #threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        #eye = cv2.resize(eye, None, fx=5, fy=5)
        # cv2.imshow("1_th_eye", threshold_eye_1)
        # cv2.imshow("th_eye_2", threshold_eye_2)
        # cv2.imshow("eye", eye)
        # cv2.imshow("left_eye", left_eye)
        # cv2.imshow("left_th", left_side_th)
        # cv2.imshow("right_th", right_side_th)
        


    cv2.imshow("Frame", frame)
    cv2.imshow("New_frame", new_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

#cap.realease()
cv2.destroyAllWindows()
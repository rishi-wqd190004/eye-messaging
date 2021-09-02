import cv2
import numpy as np

keyboard = np.zeros((1000,1500,3), np.uint8)

# print(keyboard.shape)

# create a virtual keyboard as its just rectangle blocks with alphabets inside
def letters(x,y, text):
    wt = 200
    ht = 200
    thickness = 3
    cv2.rectangle(keyboard, (x+thickness, y+thickness), (x+wt-thickness, y+ht-thickness), (0,255,255), thickness)

    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_thickness = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_thickness)[0]
    wt_text, ht_text = text_size[0], text_size[1]
    text_x = int((wt - wt_text) / 2 ) + x
    text_y = int((ht + ht_text) / 2 ) + y
    cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (0,255,255), font_thickness)



while True:
    letters(0, 0, "A")
    letters(200, 0, "B")
    letters(400, 0, "C")
    cv2.imshow("keyboard", keyboard)
    key = cv2.waitKey(1)
    if key == 27:
        break


cv2.destroyAllWindows()
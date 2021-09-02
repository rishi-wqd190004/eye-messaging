## eye-messaging
### Here we will take the eye input, split the movement and map it to on-screen keyboard. --> Main reason to create this is for sliding texting.

# Flask covered eye messaging application to perform swipe text on virtual keyboard
-----------------------------------------------------------------------------------------------
#### Note: This is a learning and a potfilo building project and many steps will be included in this like
#### Few questions which came in mind before and during the project:
 - Why not use hand gesture?
    - Reason of not using hand gesture: 
        - a. Want to explore what you can do with eye tracking
        - b. Yes you can use hand gesture but imagine being looking and able to write when your hands are cutting vegetables or cleaning your washroom (haha just being sarcastic)
    
-----------------------------------------------------------------------------------------------
#### Here I added usage of *dlib* package for getting pointer of points
## Idealogy:
One have seen swiping fingers on keyboard on your phones like below:
![Source: Google Blog AI](https://1.bp.blogspot.com/-Oz-oMYfar8I/WSW90Jo866I/AAAAAAAAB1k/GO-9rpbpTcMhsfy3edD3lgcjXlLlTQjlwCLcB/s640/image6.gif)

Use a similar approach and try to type on the screen or gaze on the keywords.

## Approach:
 - Use your laptop camera to view your eye and map it to the gaze location.
 - Search the gaze in which direction. 
 - Map it to a keyboard
 - Get the words printed

## Steps taken:
Get information as shown below screenshot using cv2.
![Mask results with cropped eye output](screenshots/Screenshot 2021-09-02 at 6.46.55 AM.png?raw=True)
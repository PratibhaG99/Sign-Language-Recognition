import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
from flask import request
from flask import jsonify
from flask import Flask
app = Flask(__name__)

def get_model():
    global model
    model = keras.models.load_model("Model2-copy1.h5")
    print("Model loaded!")
    
print("Loading Keras Model!!!")
get_model()

background=None
@app.route('/capture', methods=['POST'])
def capture():
    message = request.get_json(force=True)
    label = message['label']
    accumulated_weight = 0.5
    ROI_top = 100
    ROI_bottom = 300
    ROI_right = 150
    ROI_left = 350
    cam = cv2.VideoCapture(0)
    num_frames = 0
    num_image_taken=0

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)


        if num_frames < 100:

            cal_accum_avg(gray_frame, accumulated_weight)
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
        elif num_frames <300:
        
            hand = segment_hand(gray_frame) 

            cv2.putText(frame_copy, "Adjust Hand",(80,400),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            if hand is not None:

                threshold, hand_segment = hand
                cv2.drawContours(frame_copy,[hand_segment + (ROI_right,ROI_top)], -1,(255,0,0),2)
                cv2.imshow("Thresholded Image",threshold)
                
        elif num_frames < 310:
            hand = segment_hand(gray_frame)

            if hand is not None:
                threshold, hand_segment = hand
                cv2.drawContours(frame_copy,[hand_segment + (ROI_right,ROI_top)], -1,(255,0,0),5)

                cv2.putText(frame_copy,"DONE",(20,400),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) 

                cv2.imshow("Thresholded Image",threshold)
                if num_image_taken <1:
                        cv2.imwrite(r"C:\Users\INTEL 2022\Desktop\gesture\temp\result"+"\\" + str(num_image_taken) + '.jpg', threshold)

                else:
                    break
                num_image_taken+=1

            else:
                    cv2.putText(frame_copy, "No hand Detected....",(60,400),cv2.FONT_HERSHEY_SIMPLEX,1,(225,255,0),2)
          
        else:
            break
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)

        # incrementing the number of frames for tracking
        num_frames += 1

        # Display the frame with segmented hand
        cv2.putText(frame_copy, "Hand sign recognition!!!",(30,90),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("Sign Language Detection",frame_copy)

        # Close windows with Esc
        
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    # Release the camera and destroy all the windows
    cam.release()
    cv2.destroyAllWindows()
    st=""
#    word_dict = {0:'ka', 1:'kha', 2:'ga', 3:'gha', 4:'cha', 5:'chha', 6:'ja', 7:'jha', 8:'ta', 9:'tha', 10:'da', 11:'dha'}
    path=r"C:\Users\INTEL 2022\Desktop\gesture\temp"
    img = ImageDataGenerator().flow_from_directory(directory=path,
                                        target_size=(64,64),class_mode='categorical', batch_size=1)
    predictions= model.predict_generator(img, verbose=0)
    for ind, i in enumerate(predictions):
        t= np.argmax(i) + 1
    
    if label==t:
        response = {
            'prediction':{
                'answer':'GOOD ONE...YOU DID RIGHT',
                'image':'1'
            }
        }
    else:
        response = {
            'prediction':{
                'answer':'OPPSS...PLEASE TRY AGAIN',
                'image':'0'
            }
        }
 
    return jsonify(response)

def cal_accum_avg(frame, accumulated_weight):

    global background
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)
    

def segment_hand(frame, threshold=25):
    global background   
    diff = cv2.absdiff(background.astype("uint8"), frame)

    
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    #Fetching contours in the frame (of hand)
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        # Returning the hand segment(max contour) and the thresholded image of hand
        return (thresholded, hand_segment_max_cont)
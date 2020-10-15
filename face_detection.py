import cv2
import winsound  

# frequency is set to 500Hz 
freq = 500 
  
# duration is set to 100 milliseconds              
dur = 100
cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
i=0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.waitKey(300)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, 0)
    detections = cascade_classifier.detectMultiScale(gray,1.3,5)

    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) 
        # save the detected frame
        cv2.imwrite('detected_face/detected'+str(i)+'.jpg',frame)
        # beep when frame detect
        winsound.Beep(freq, dur) 
        
    # Display the resulting frame
    cv2.imshow('frame',frame)
    #cv2.imwrite('detected'+str(i)+'.jpg',frame)
    i=i+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
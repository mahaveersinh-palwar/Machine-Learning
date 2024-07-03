import cv2

img= cv2.imread("2.jpg", cv2.IMREAD_COLOR)
# fc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
fc = cv2.CascadeClassifier("haarcascade_eye.xml")
# fc = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
# fc = cv2.CascadeClassifier("haarcascade_mcs_rightear.xml")
# fc = cv2.CascadeClassifier("haarcascade_mcs_leftear.xml")

grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = fc.detectMultiScale(grey_image, 1.1 ,4)
print(faces)

for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
    
    cv2.imshow("face Detection",img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

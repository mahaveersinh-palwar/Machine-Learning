import cv2

# Load the image
img = cv2.imread("images.jpg", cv2.IMREAD_COLOR)

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)

# Draw circles on the detected faces
for (x, y, w, h) in faces:
    # Calculate the center of the face
    cx = x + w // 2
    cy = y + h // 2
    center = (cx, cy)
    
    # Radius of the circle
    radius = min(w, h) // 2
    
    # Draw the circle on the face
    cv2.circle(img, center, radius, (0, 255, 0), 3)

# Display the image with circles drawn on the faces
cv2.imshow("Detected Faces with Circles", img)

# Wait for a key press and close the window after 5000 milliseconds (5 seconds)
cv2.waitKey(5000)

# Close all OpenCV windows
cv2.destroyAllWindows()

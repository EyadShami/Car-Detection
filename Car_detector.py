import cv2

# Load the car cascade
car_cascade = cv2.CascadeClassifier('cars.xml')

# Open the video file
cap = cv2.VideoCapture('Cars.mp4')

# Create a window
window_name = "Car Detection"
cv2.namedWindow(window_name)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    # Draw rectangles around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow(window_name, frame)

    # Close if 'q' is pressed or if the window is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()

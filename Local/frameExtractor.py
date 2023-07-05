import cv2
import os

face_cascade = cv2.CascadeClassifier('Masked-Face-Recognition-from-CCTV-Footage/Local/frameExtractor.py')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

video_path = 'video.mp4'
output_dir = 'outputs/'

cap = cv2.VideoCapture(video_path)
face_count = 0
face_files = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = detect_faces(frame)
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_filename = f'face_{face_count}.jpg'
        face_path = os.path.join(output_dir, face_filename)
        
        # Store the face image
        cv2.imwrite(face_path, face_img)
        
        # Keep track of unique face files
        face_files.add(face_filename)
        
        face_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"Found {len(face_files)} distinct faces.")

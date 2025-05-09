import cv2
import mediapipe as mp
from PIL import Image

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# List of frames for GIF
frames = []

#The Rotate video output
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated


# Start processing the live video
cap = cv2.VideoCapture("C:/Users/fayzi-dev/Desktop/ComputerVision/my_face_video.mp4")
with mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Number of faces 
    refine_landmarks=True,  # For more precise landmarks on eyes and lips 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    target_size = (640, 360)
    while cap.isOpened():
      

        ret, frame = cap.read()
        if not ret:
            break


        # Convert BGR to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # Draw points
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,  # To draw the complete face mesh
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
        frame = cv2.resize(frame, target_size) 
        frame = rotate_image(frame, 90)  
        # Convert frame to PIL.Image and save to list
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(pil_image)


        cv2.imshow('Face Mesh', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# print(f"Total frames to save: {len(frames)}")
# if len(frames) > 1:
#     frames[0].save('face_output.gif', save_all=True,
#                    append_images=frames[1:], duration=100, loop=0)
# else:
#     print("Not enough frames to save GIF.")


# Save GIF file with Pillow
pil_image = pil_image.resize((640, 360))
if frames:
    frames[0].save(
        'face_output.gif',
        save_all=True,
        append_images=frames[1:],
        duration=20, 
        loop=0  
    )

import cv2
import mediapipe as mp
import os

def extract_lip_frames(
    video_path="data/raw_videos/namaste.mp4",
    output_folder="data/lip_frames/namaste.mp4"
    ):
    os.makedirs(output_folder, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh

    # Initialize FaceMesh with improved settings
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,        
        refine_landmarks=True,   
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    ) as face_mesh:
        cap = cv2.VideoCapture(video_path)
        frame_num = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            # Debug: Check if face was detected
            if results.multi_face_landmarks:
                print(f"Face detected at frame {frame_num}")
                landmarks = results.multi_face_landmarks[0]  # Only take the first detected face
                lip_points = [landmarks.landmark[i] for i in range(61, 88)]  # Lip landmarks

                xs = [p.x for p in lip_points]
                ys = [p.y for p in lip_points]

                h, w, _ = frame.shape
                x_min, x_max = int(min(xs) * w), int(max(xs) * w)
                y_min, y_max = int(min(ys) * h), int(max(ys) * h)

                # Add some padding to the lip region
                padding = 5
                x_min = max(0, x_min - padding)
                x_max = min(w, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(h, y_max + padding)

                lip_crop = frame[y_min:y_max, x_min:x_max]

                # Only save the frame if the crop is not empty
                if lip_crop.size != 0:
                    lip_crop = cv2.resize(lip_crop, (100, 50))  # Resize to standard size (100x50)
                    save_path = os.path.join(output_folder, f"frame_{frame_num:04d}.jpg")
                    cv2.imwrite(save_path, lip_crop)
                    frame_num += 1

            else:
                print(f"No face detected in frame {frame_num}")

        cap.release()
    print(f"Lip frames saved to {output_folder}")

if __name__ == "__main__":
    # Set video path and output folder
    video_path = "data/raw_videos/namaste.mp4"  # Change to your video path
    output_folder = "data/lip_frames/namaste"   # Change to your desired output folder

    # Extract lip frames from the video
    extract_lip_frames(video_path, output_folder)

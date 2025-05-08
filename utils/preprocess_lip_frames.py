import cv2
import os
import numpy as np

def preprocess_lip_frames(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):  # Only process .jpg images
            frame_path = os.path.join(input_folder, filename)
            frame = cv2.imread(frame_path)

            # Normalize the image (scale pixel values between 0 and 1)
            frame = frame / 255.0

            # Convert the frame back to uint8 before processing further
            frame = (frame * 255).astype(np.uint8)

            # Convert the frame to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize frame to a fixed size if needed (optional)
            frame = cv2.resize(frame, (100, 50))

            # Save the preprocessed frame in the output folder
            preprocessed_frame_path = os.path.join(output_folder, filename)
            cv2.imwrite(preprocessed_frame_path, frame)  # Save as uint8 image

    print(f"Frames have been preprocessed and saved to {output_folder}")

if __name__ == "__main__":
    input_folder = "data/lip_frames/namaste"  # Folder with extracted lip frames
    output_folder = "data/lip_frames_processed/namaste"  # Folder to save preprocessed frames

    preprocess_lip_frames(input_folder, output_folder)

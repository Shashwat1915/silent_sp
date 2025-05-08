import cv2
import os
import time

def record_video(save_dir="data/raw_videos"):
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        return

    print("Press 's' to start recording.")
    print("Press 'q' to stop recording.")

    recording = False
    frames = []
    fps = 20  # Frames per second
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        cv2.imshow("Recording Window", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and not recording:
            print("Started recording...")
            recording = True
            frames = []

        elif key == ord('q') and recording:
            print("Stopped recording.")
            recording = False

            # Ask for filename
            filename = input("Enter name for the saved video (without extension): ").strip()
            full_path = os.path.join(save_dir, f"{filename}.mp4")

            # Save video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(full_path, fourcc, fps, (width, height))

            for f in frames:
                out.write(f)

            out.release()
            print(f"Video saved at: {full_path}")

        if recording:
            frames.append(frame)

        # Press 'ESC' key to exit completely
        if key == 27:
            print("Exiting camera...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    record_video()

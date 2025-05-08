import csv

# Frame list (128 frames)
frames = [f"frame_{i:04d}.jpg" for i in range(1, 200)]  # frame_0001.jpg to frame_0127.jpg

# Word label
label = 'hello'

# Create CSV
with open('frame_labels2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['frames', 'label'])  # Header
    frames_str = ' '.join(frames)  # Join all frames with space
    writer.writerow([frames_str, label])

print("CSV created successfully!")

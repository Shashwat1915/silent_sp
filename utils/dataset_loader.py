from torch.utils.data import Dataset
import cv2  # OpenCV to read images
import torch

class LipReadingDataset(Dataset):
    def __init__(self, csv_file, frames_root, transform=None):
        self.data = []
        self.frames_root = frames_root
        self.transform = transform

        # Read the CSV
        with open(csv_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                frames, label = line.strip().split(',')
                frame_list = frames.split()
                self.data.append((frame_list, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_list, label = self.data[idx]

        frames = []
        for frame_name in frame_list:
            frame_path = f"{self.frames_root}/{frame_name}"
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        frames = torch.stack(frames)  # Shape: (num_frames, C, H, W)
        
        return frames, label

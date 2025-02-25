import cv2
import os
import glob

def create_video_from_images(image_folder, output_video, fps=30, duration_per_image=4, codec='mp4v'):
    # Get list of image paths, sorted by filename.
    image_paths = sorted(glob.glob(os.path.join(image_folder, '*.*')))
    
    if not image_paths:
        print("No images found in the folder.")
        return

    # Read the first image to get the frame dimensions.
    frame = cv2.imread(image_paths[0])
    if frame is None:
        print("Error reading the first image.")
        return

    height, width, _ = frame.shape
    frame_size = (width, height)

    # Define the codec and create VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
    
    # Calculate the number of frames each image should appear.
    frames_per_image = int(fps * duration_per_image)
    
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Error reading image {img_path}, skipping.")
            continue
        
        # Resize frame if needed (here we ensure all frames have the same size)
        frame = cv2.resize(frame, frame_size)
        
        for _ in range(frames_per_image):
            out.write(frame)
        print(f"Added {img_path} to video for {duration_per_image} seconds.")

    out.release()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    # Path to the folder containing images.
    image_folder = "/Users/esbenchristensen/Github/lpr/images"
    # Output video file.
    output_video = "output_video.mp4"
    # Frames per second.
    fps = 30
    # Duration per image in seconds.
    duration_per_image = 4
    create_video_from_images(image_folder, output_video, fps, duration_per_image)
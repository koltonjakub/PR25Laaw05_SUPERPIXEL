import cv2
import os

# Base path and filename template
base_path = 'images'
filename_template = 'superpixel_regions_iter_{}.jpg'

# Number of frames to visualize
num_frames = 10

for i in range(1, num_frames + 1):
    file_path = os.path.join(base_path, filename_template.format(i))
    
    # Load image
    image = cv2.imread(file_path)
    image = cv2.resize(image, (640, 480))  # Resize for better visualization
    if image is None:
        print(f"Warning: Could not load image {file_path}")
        continue

    # Display the image
    cv2.imshow(f'Frame ', image)
    key = cv2.waitKey(500)  # Wait 500ms between frames, or press any key to continue

    # Optional: close the previous window
    #cv2.destroyWindow(f'Frame {i}')

# Clean up
cv2.destroyAllWindows()

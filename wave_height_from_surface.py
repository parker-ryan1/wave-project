import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'practice wave.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
    exit(1)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find the middle column of the image
middle_x = img.shape[1] // 2

# Create a region of interest around the middle
roi_width = 40  # pixels on each side of the middle
roi = edges[:, middle_x-roi_width:middle_x+roi_width]

# Sum across the ROI width to get a stronger signal
edge_profile = np.sum(roi, axis=1)
edge_profile = edge_profile / np.max(edge_profile) * 255  # Normalize for visualization

# Find significant edges (wave peaks and the water surface)
significant_edges = []
for i in range(1, len(edge_profile)-1):
    # Look for peaks in the edge profile
    if edge_profile[i] > 50 and edge_profile[i] > edge_profile[i-1] and edge_profile[i] > edge_profile[i+1]:
        significant_edges.append(i)

# Sort edges by position (top to bottom)
significant_edges.sort()

if len(significant_edges) >= 2:
    # Assume the water surface is the lowest significant edge in the upper half of the image
    upper_half = [edge for edge in significant_edges if edge < img.shape[0]/2]
    if upper_half:
        water_surface_y = max(upper_half)
    else:
        # If no edge in upper half, take the highest edge
        water_surface_y = min(significant_edges)
    
    # Find the middle wave peak (should be one of the prominent edges)
    # We'll look for edges below the water surface
    wave_candidates = [edge for edge in significant_edges if edge > water_surface_y]
    
    if wave_candidates:
        # Take the first peak below the water surface as the wave peak
        wave_peak_y = wave_candidates[0]
        
        # Calculate wave height
        wave_height_pixels = wave_peak_y - water_surface_y
        print(f"Estimated wave height from surface: {wave_height_pixels} pixels")
        
        # Visualize the results
        result_img = img.copy()
        
        # Draw water surface line
        cv2.line(result_img, (middle_x - 100, water_surface_y), (middle_x + 100, water_surface_y), (0, 255, 0), 2)
        cv2.putText(result_img, "Water Surface", (middle_x - 200, water_surface_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw wave peak and height measurement
        cv2.line(result_img, (middle_x, water_surface_y), (middle_x, wave_peak_y), (0, 0, 255), 2)
        cv2.putText(result_img, f"Wave Height: {wave_height_pixels}px", 
                   (middle_x + 10, (water_surface_y + wave_peak_y) // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(231)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        plt.subplot(232)
        plt.title('Edge Detection')
        plt.imshow(edges, cmap='gray')
        
        plt.subplot(233)
        plt.title('Middle Region ROI')
        plt.imshow(roi, cmap='gray')
        
        plt.subplot(234)
        plt.title('Wave Height Measurement')
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        
        plt.subplot(235)
        plt.title('Edge Profile')
        plt.plot(edge_profile, range(len(edge_profile)))
        plt.axhline(y=water_surface_y, color='g', linestyle='-', alpha=0.5)
        plt.axhline(y=wave_peak_y, color='r', linestyle='-', alpha=0.5)
        for edge in significant_edges:
            plt.axhline(y=edge, color='b', linestyle='--', alpha=0.3)
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        
        plt.tight_layout()
        plt.savefig('wave_height_from_surface_result.png')
        plt.show()
    else:
        print("Could not detect wave peak below the water surface")
else:
    print("Could not detect enough significant edges in the image") 
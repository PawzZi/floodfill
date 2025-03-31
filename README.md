# floodfill

floodfill - akin to bluebeam one

ff1

import numpy as np
import cv2
from collections import deque
import random

def generate_distinct_colors(n):
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        # Generate HSV colors with good separation
        hue = (i * 137.5) % 360  # Golden angle in degrees
        saturation = 75 + random.random() * 25  # 75-100%
        value = 75 + random.random() * 25  # 75-100%
        
        # Convert HSV to BGR
        hsv_color = np.uint8([[[hue, saturation, value]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        
        colors.append(bgr_color.tolist())
    return colors

def flood_fill_zones(segmentation_mask):
    """
    Perform flood fill from each exit point, creating distinct zones.
    
    Args:
        segmentation_mask: numpy array with BGR colors:
            - Exit: RGB(51,222,87)
            - Obstruction: RGB(152,80,164)
            - Obstruction Variable: RGB(42,216,205)
            - Background: RGB(0,0,0)
    
    Returns:
        zone_mask: numpy array where each zone has a unique ID
        colored_visualization: BGR image showing the zones in different colors
    """
    height, width = segmentation_mask.shape[:2]
    
    # Define the label colors in BGR format
    exit_color = np.array([87, 222, 51])  # BGR format
    obstruction_color = np.array([164, 80, 152])
    obstruction_var_color = np.array([205, 216, 42])
    
    # Create masks for different elements
    exits_mask = np.all(segmentation_mask == exit_color, axis=2)
    obstructions_mask = np.logical_or(
        np.all(segmentation_mask == obstruction_color, axis=2),
        np.all(segmentation_mask == obstruction_var_color, axis=2)
    )
    
    # First, label connected exit regions
    num_labels, exit_regions = cv2.connectedComponents(exits_mask.astype(np.uint8))
    
    # Initialize zone mask
    zone_mask = np.zeros((height, width), dtype=np.int32)
    zone_mask[obstructions_mask] = -1  # Mark obstructions with -1
    
    # Generate distinct colors for visualization (make them slightly transparent for better visibility)
    colors = generate_distinct_colors(num_labels - 1)
    colors = [[int(c * 0.7) for c in color] for color in colors]  # Make zone colors more subtle
    colored_visualization = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Collect all edge pixels for each zone first
    edge_pixels = [[] for _ in range(num_labels)]
    
    for zone_id in range(1, num_labels):
        # Find all pixels in this exit region
        region_pixels = np.where(exit_regions == zone_id)
        if len(region_pixels[0]) == 0:
            continue
        
        # Mark all pixels of this exit with the zone_id
        zone_mask[region_pixels] = zone_id
        # Keep original exit color
        colored_visualization[region_pixels] = exit_color
        
        # Find edge pixels (pixels with at least one unlabeled neighbor)
        for y, x in zip(region_pixels[0], region_pixels[1]):
            is_edge = False
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_y, new_x = y + dy, x + dx
                if (0 <= new_y < height and 0 <= new_x < width and
                    zone_mask[new_y, new_x] == 0 and
                    not obstructions_mask[new_y, new_x] and
                    np.all(segmentation_mask[new_y, new_x] == [0, 0, 0])):
                    is_edge = True
                    edge_pixels[zone_id].append((new_y, new_x))
                    
            if is_edge:
                edge_pixels[zone_id].append((y, x))
    
    # Create a queue with all edge pixels from all zones
    master_queue = deque()
    for zone_id in range(1, num_labels):
        for pixel in edge_pixels[zone_id]:
            master_queue.append((pixel[0], pixel[1], zone_id))
    
    # Perform flood fill from all edge pixels simultaneously
    while master_queue:
        y, x, zone_id = master_queue.popleft()
        current_color = colors[zone_id - 1]
        
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_y, new_x = y + dy, x + dx
            
            # Check bounds
            if not (0 <= new_y < height and 0 <= new_x < width):
                continue
            
            # Check if:
            # 1. Cell is unassigned (zone_mask == 0)
            # 2. Not an obstruction (zone_mask != -1)
            # 3. Is background color (black)
            if (zone_mask[new_y, new_x] == 0 and
                not obstructions_mask[new_y, new_x] and
                np.all(segmentation_mask[new_y, new_x] == [0, 0, 0])):
                
                zone_mask[new_y, new_x] = zone_id
                colored_visualization[new_y, new_x] = current_color
                master_queue.append((new_y, new_x, zone_id))
    
    # Make sure obstructions are visible in the visualization
    colored_visualization[obstructions_mask] = [0, 0, 0]  # Black for obstructions
    
    # Ensure exits are clearly visible by reapplying their original color
    colored_visualization[exits_mask] = exit_color
    
    return zone_mask, colored_visualization

def process_image(image_path):
    """
    Process the segmentation mask and create zone visualization.
    
    Args:
        image_path: Path to the segmentation mask image
    
    Returns:
        zone_mask: Array with zone IDs
        visualization: Colored visualization of zones
    """
    # Read the segmentation mask in BGR format
    segmentation_mask = cv2.imread(image_path)
    
    if segmentation_mask is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Process the zones
    zone_mask, visualization = flood_fill_zones(segmentation_mask)
    
    return zone_mask, visualization

if __name__ == "__main__":
    # Example usage
    image_path = r"ETD Project/Test Case - Car Parks/Labeled/test1.png"
    zone_mask, visualization = process_image(image_path)
    
    # Save visualization in the same folder as input
    output_path = r"ETD Project/Test Case - Car Parks/Labeled/zone_visualization.png"
    cv2.imwrite(output_path, visualization)
    print(f"Visualization saved to: {output_path}")
    print(f"Number of zones identified: {np.max(zone_mask)}")
    
    # Print detailed information
    print("\nZone Information:")
    for zone_id in range(1, np.max(zone_mask) + 1):
        zone_pixels = np.sum(zone_mask == zone_id)
        print(f"Zone {zone_id}:")
        print(f"  - Area covered: {zone_pixels} pixels")
    
    # Print any unassigned areas (excluding obstructions)
    unassigned = np.sum((zone_mask == 0) & (zone_mask != -1))
    print(f"\nUnassigned walkable area: {unassigned} pixels")
    
    # Print obstruction information
    obstructions = np.sum(zone_mask == -1)
    print(f"Total obstruction area: {obstructions} pixels") 

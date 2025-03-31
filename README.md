# floodfill

floodfill - akin to bluebeam one

ff1 - Pixel Based

import numpy as np
import cv2
import heapq
import os

def generate_distinct_colors(n):
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 128, 0],  # Orange
        [128, 0, 255],  # Purple
        [0, 255, 128],  # Spring Green
        [255, 0, 128]   # Pink
    ]
    
    if n <= len(colors):
        return np.array(colors[:n], dtype=np.uint8)
    
    additional_colors = []
    for i in range(len(colors), n):
        hue = (i * 137.5) % 360
        hsv_color = np.array([[[hue, 95, 95]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        additional_colors.append(bgr_color)
    
    return np.array(colors + additional_colors, dtype=np.uint8)

def upscale_mask(mask, scale_factor):
    """Upscale the mask using nearest neighbor interpolation to preserve exact colors."""
    height, width = mask.shape[:2]
    new_height, new_width = height * scale_factor, width * scale_factor
    return cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

def downscale_mask(mask, original_shape):
    """Downscale the mask using area interpolation for smoother results."""
    return cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_AREA)

def flood_fill_zones(segmentation_mask):
    """
    Create zones using simultaneous expansion from all exits.
    """
    height, width = segmentation_mask.shape[:2]
    
    # Define colors
    exit_color = np.array([87, 222, 51], dtype=np.uint8)
    obstruction_color = np.array([164, 80, 152], dtype=np.uint8)
    obstruction_var_color = np.array([205, 216, 42], dtype=np.uint8)
    
    # Create masks
    exits_mask = np.all(segmentation_mask == exit_color, axis=2)
    obstructions_mask = np.any(
        [np.all(segmentation_mask == color, axis=2) 
         for color in [obstruction_color, obstruction_var_color]], 
        axis=0
    )
    
    # Label connected exit regions
    num_labels, exit_regions = cv2.connectedComponents(exits_mask.astype(np.uint8))
    
    # Initialize masks
    zone_mask = np.zeros((height, width), dtype=np.int32)
    zone_mask[obstructions_mask] = -1
    distance_mask = np.full((height, width), np.inf)
    
    # Generate colors
    colors = generate_distinct_colors(num_labels - 1)
    colored_visualization = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Initialize master queue for all zones
    master_queue = []
    
    # Directions for 8-connectivity with weights
    directions = [
        (-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),
        (0, -1, 1.0),                        (0, 1, 1.0),
        (1, -1, np.sqrt(2)),   (1, 0, 1.0),  (1, 1, np.sqrt(2))
    ]
    
    # Initialize from all exits simultaneously
    for zone_id in range(1, num_labels):
        region_pixels = np.where(exit_regions == zone_id)
        if len(region_pixels[0]) == 0:
            continue
            
        # Mark exits
        zone_mask[region_pixels] = zone_id
        colored_visualization[region_pixels] = exit_color
        
        # Add edge pixels to master queue
        for y, x in zip(region_pixels[0], region_pixels[1]):
            for dy, dx, weight in directions:
                new_y, new_x = int(y + dy), int(x + dx)
                if (0 <= new_y < height and 0 <= new_x < width and
                    zone_mask[new_y, new_x] == 0 and
                    not obstructions_mask[new_y, new_x]):
                    # Queue format: (distance, y, x, zone_id)
                    heapq.heappush(master_queue, (weight, new_y, new_x, zone_id))
                    distance_mask[new_y, new_x] = weight
    
    # Process all zones simultaneously
    while master_queue:
        dist, y, x, zone_id = heapq.heappop(master_queue)
        
        # Skip if already assigned
        if zone_mask[y, x] != 0:
            continue
        
        # Assign zone
        zone_mask[y, x] = zone_id
        colored_visualization[y, x] = colors[zone_id - 1]
        
        # Add neighbors to queue
        for dy, dx, weight in directions:
            new_y, new_x = int(y + dy), int(x + dx)
            new_dist = dist + weight
            
            if (0 <= new_y < height and 0 <= new_x < width and
                zone_mask[new_y, new_x] == 0 and
                not obstructions_mask[new_y, new_x] and
                new_dist < distance_mask[new_y, new_x]):
                
                distance_mask[new_y, new_x] = new_dist
                heapq.heappush(master_queue, (new_dist, new_y, new_x, zone_id))
    
    # Final visualization
    colored_visualization[obstructions_mask] = [0, 0, 0]
    colored_visualization[exits_mask] = exit_color
    
    return zone_mask, colored_visualization

def process_image(image_path):
    """Process the segmentation mask and create zone visualization."""
    segmentation_mask = cv2.imread(image_path)
    if segmentation_mask is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    return flood_fill_zones(segmentation_mask)

if __name__ == "__main__":
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Construct paths
    image_path = os.path.join(project_dir, "ETD Project", "Test Case - Car Parks", "Labeled", "test1.png")
    output_path = os.path.join(project_dir, "ETD Project", "Test Case - Car Parks", "Labeled", "zone_visualization.png")
    
    print(f"Reading image from: {image_path}")
    zone_mask, visualization = process_image(image_path)
    
    print(f"Saving visualization to: {output_path}")
    cv2.imwrite(output_path, visualization)
    print(f"Number of zones identified: {np.max(zone_mask)}")
    
    # Print zone information
    print("\nZone Information:")
    unique_zones = np.unique(zone_mask[zone_mask > 0])
    for zone_id in unique_zones:
        zone_pixels = np.sum(zone_mask == zone_id)
        print(f"Zone {zone_id}: {zone_pixels} pixels")
    
    # Print statistics
    unassigned = np.sum((zone_mask == 0) & (zone_mask != -1))
    obstructions = np.sum(zone_mask == -1)
    print(f"\nUnassigned walkable area: {unassigned} pixels")
    print(f"Total obstruction area: {obstructions} pixels") 

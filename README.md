# floodfill
# better one ish




import numpy as np
import cv2
import os
import json
import base64
import struct
import heapq
import math
from multiprocessing import Pool, cpu_count

def get_min_exit_width(annotations) -> float:
    """Get the minimum width of all exits to use as reference for passable spaces."""
    min_width = float('inf')
    for annotation in annotations:
        if annotation['category_id'] == 3:  # Exit
            bbox = annotation['bbox']
            width = bbox[2]  # width is the third value in bbox
            min_width = min(min_width, width)
    return min_width

def get_min_exit_height(annotations) -> float:
    """Get the minimum height of all exits to use as reference for passable spaces."""
    min_height = float('inf')
    for annotation in annotations:
        if annotation['category_id'] == 3:  # Exit
            bbox = annotation['bbox']
            height = bbox[3]  # height is the fourth value in bbox
            min_height = min(min_height, height)
    print(f"Found exit with height: {height}")
    return min_height

def get_max_door_width(annotations) -> float:
    """Get the maximum door width by calculating the maximum distance between any two points in door polygons."""
    max_width = 0
    
    for annotation in annotations:
        if annotation['category_id'] == 3 and 'segmentation' in annotation:  # Exit
            if isinstance(annotation['segmentation'], list) and len(annotation['segmentation']) > 0:
                # Convert segmentation points to numpy array of x,y coordinates
                points = np.array(annotation['segmentation'][0]).reshape(-1, 2)
                
                # Calculate distances between all pairs of points
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        width = np.linalg.norm(points[i] - points[j])
                        max_width = max(max_width, width)
                
                print(f"Found door with max width: {max_width}")
    
    return max_width

def find_narrow_passages(binary_img: np.ndarray, max_door_width: float, debug_dir: str) -> list:
    """Find narrow passages and return them as vector annotations."""
    # Calculate the narrow passage threshold as 60% of the maximum door width
    narrow_threshold = int(max_door_width * 0.6)
    print(f"Using narrow passage threshold: {narrow_threshold} pixels (60% of max door width {max_door_width})")
    
    # Invert the binary image so obstacles are white (255)
    obstacles = cv2.bitwise_not(binary_img)
    
    # Create kernels for horizontal and vertical narrow passages
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (narrow_threshold, 5))  # Increased from 3 to 5
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, narrow_threshold))    # Increased from 3 to 5
    
    # Process horizontal narrow passages
    horizontal_dilated = cv2.dilate(obstacles, horizontal_kernel, iterations=2)  # Increased from 1 to 2
    horizontal_eroded = cv2.erode(horizontal_dilated, horizontal_kernel, iterations=2)    # Increased from 1 to 2
    horizontal_narrow = cv2.bitwise_xor(obstacles, horizontal_eroded)
    
    # Process vertical narrow passages
    vertical_dilated = cv2.dilate(obstacles, vertical_kernel, iterations=2)     # Increased from 1 to 2
    vertical_eroded = cv2.erode(vertical_dilated, vertical_kernel, iterations=2)        # Increased from 1 to 2
    vertical_narrow = cv2.bitwise_xor(obstacles, vertical_eroded)
    
    # Combine horizontal and vertical narrow passages
    narrow_passages = cv2.bitwise_or(horizontal_narrow, vertical_narrow)
    
    # Add a small dilation to connect nearby passages and fill small gaps
    connect_kernel = np.ones((3, 3), np.uint8)
    narrow_passages = cv2.dilate(narrow_passages, connect_kernel, iterations=1)
    narrow_passages = cv2.erode(narrow_passages, connect_kernel, iterations=1)
    
    # Find contours in the narrow passages
    contours, _ = cv2.findContours(narrow_passages, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert contours to COCO-style annotations
    narrow_annotations = []
    for i, contour in enumerate(contours):
        # Flatten the contour points into a 1D list [x1, y1, x2, y2, ...]
        segmentation = contour.flatten().tolist()
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Create annotation dict
        annotation = {
            'id': i + 1000,  # Start IDs from 1000 to avoid conflicts
            'category_id': 4,  # New category for narrow passages
            'bbox': [x, y, w, h],
            'segmentation': [segmentation],
            'area': cv2.contourArea(contour)
        }
        narrow_annotations.append(annotation)
    
    print(f"Found {len(narrow_annotations)} narrow passage regions")
    return narrow_annotations

def find_nearest_free_pixel(binary: np.ndarray, cx: int, cy: int) -> tuple:
    """Find the nearest free (passable) pixel to the given coordinates."""
    height, width = binary.shape
    min_dist = float('inf')
    best_point = None
    
    # Search in expanding squares around the centroid
    for r in range(1, 50):  # Limit search radius to avoid excessive searching
        # Check all pixels in a square of radius r around the centroid
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                ny, nx = cy + dy, cx + dx
                
                # Check bounds and if pixel is passable
                if (0 <= ny < height and 0 <= nx < width and binary[ny, nx] > 0):
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_point = (ny, nx)
        
        # If we found a point, return it
        if best_point is not None:
            print(f"Found nearest free pixel at {best_point}, {min_dist:.1f} pixels from centroid")
            return best_point
    
    raise ValueError(f"No free pixel found near ({cx}, {cy})")

def calculate_obstacle_proximity(binary: np.ndarray, cy: int, cx: int, max_check: int = 5) -> float:
    """Calculate how close the current point is to obstacles."""
    height, width = binary.shape
    obstacle_distance = float('inf')
    
    # Check in a small radius around the point
    for dy in range(-max_check, max_check + 1):
        for dx in range(-max_check, max_check + 1):
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < height and 0 <= nx < width and 
                binary[ny, nx] == 0):  # Found obstacle
                dist = np.sqrt(dy*dy + dx*dx)
                obstacle_distance = min(obstacle_distance, dist)
    
    return obstacle_distance

def create_obstacle_distance_map(binary: np.ndarray) -> np.ndarray:
    """Precompute obstacle distances for all points using distance transform."""
    # Invert binary image so obstacles are 0 and free space is 1
    inverted = cv2.bitwise_not(binary) // 255
    # Calculate distance to nearest obstacle
    return cv2.distanceTransform(inverted, cv2.DIST_L2, 5)

def calculate_path_segment(binary: np.ndarray, cy: int, cx: int, ny: int, nx: int, 
                         look_ahead: int = 5) -> tuple[float, list]:
    """Calculate cost and path for a multi-pixel segment looking ahead."""
    height, width = binary.shape
    
    # Calculate direction vector
    dy, dx = ny - cy, nx - cx
    direction = np.array([dy, dx], dtype=float)
    direction = direction / np.linalg.norm(direction)
    
    # Calculate perpendicular vector for width checking
    perp = np.array([-direction[1], direction[0]])
    
    # Store points and their costs
    path_points = []
    total_cost = 0
    current_y, current_x = cy, cx
    
    for step in range(look_ahead):
        # Look at points perpendicular to our direction
        best_point = None
        best_cost = float('inf')
        
        # Check points in a corridor perpendicular to our direction
        for offset in range(-3, 4):  # Check 7 points perpendicular to direction
            check_y = int(current_y + perp[0] * offset)
            check_x = int(current_x + perp[1] * offset)
            
            if (0 <= check_y < height and 0 <= check_x < width and 
                binary[check_y, check_x] > 0):
                # Calculate cost based on:
                # 1. Distance from ideal straight line
                # 2. Distance from obstacles
                # 3. Progress toward goal
                line_dist = abs(offset)
                goal_progress = ((check_y - ny) ** 2 + (check_x - nx) ** 2) ** 0.5
                
                cost = line_dist + goal_progress
                if cost < best_cost:
                    best_cost = cost
                    best_point = (check_y, check_x)
        
        if best_point is None:
            break
        
        # Add best point to our path
        path_points.append(best_point)
        total_cost += best_cost
        
        # Move forward along direction
        current_y = int(current_y + direction[0])
        current_x = int(current_x + direction[1])
        
        # Stop if we've reached or passed our target
        if (current_y - ny) ** 2 + (current_x - nx) ** 2 <= 1:
            break
    
    return total_cost, path_points

def create_exit_zones(final_visualization: np.ndarray, debug_dir: str) -> list:
    """Create vector-based exit zones using a single parallel distance map."""
    height, width = final_visualization.shape[:2]
    binary = cv2.cvtColor(final_visualization, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(binary, 128, 255, cv2.THRESH_BINARY)[1]
    
    # Create binary mask (0 for obstacles, 255 for passable space)
    exits_mask = cv2.inRange(final_visualization, (0, 255, 0), (0, 255, 0))
    
    # Find connected components for exits
    num_exits, exit_labels, stats, centroids = cv2.connectedComponentsWithStats(exits_mask)
    stats = stats[1:]  # Skip background
    centroids = centroids[1:]
    num_actual_exits = len(stats)
    print(f"Found {num_actual_exits} exits")
    
    # Define number of points to sample from each exit
    POINTS_PER_EXIT = 20  # Adjust this value based on your needs
    
    # Collect all exit points with their exit IDs
    all_exit_points = []
    exit_point_ids = []
    
    for i, (stat, centroid) in enumerate(zip(stats, centroids)):
        x, y, w, h = stat[0], stat[1], stat[2], stat[3]
        print(f"\nProcessing exit {i+1} at ({x}, {y}) size {w}x{h}")
        
        # Get exit points
        exit_region = exits_mask[y:y+h, x:x+w] > 0
        if np.any(exit_region):
            exit_coords = np.where(exit_region)
            all_points = list(zip(exit_coords[0] + y, exit_coords[1] + x))
            
            # If we have more points than needed, sample evenly
            if len(all_points) > POINTS_PER_EXIT:
                # Calculate step size to sample evenly
                step = len(all_points) / POINTS_PER_EXIT
                indices = [int(j * step) for j in range(POINTS_PER_EXIT)]
                exit_points = [all_points[j] for j in indices]
            else:
                exit_points = all_points
            
            print(f"Sampled {len(exit_points)} points from {len(all_points)} total exit points")
            
            # Add points with their exit ID
            all_exit_points.extend(exit_points)
            exit_point_ids.extend([i] * len(exit_points))
    
    # Create a single distance map for all exits
    dist_map, exit_map = create_parallel_distance_map(binary, all_exit_points, exit_point_ids)
    
    # Save debug visualization for each exit
    for i in range(num_actual_exits):
        # Create a visualization for this exit's distance
        exit_dist = np.copy(dist_map)
        exit_dist[exit_map != i] = np.inf  # Only show distances for this exit
        exit_dist[exit_dist == np.inf] = 0
        exit_dist = cv2.normalize(exit_dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, f'distance_map_exit_{i+1}.png'), exit_dist)
    
    # Create zone map
    zone_map = np.zeros((height, width), dtype=np.int32)
    zone_map[binary == 0] = -1  # Mark obstacles
    
    # Assign zones based on minimum distance
    valid_mask = binary > 0
    reachable = dist_map < np.inf
    
    # Update zone map
    for i in range(num_actual_exits):
        zone_mask = (exit_map == i) & valid_mask & reachable
        zone_map[zone_mask] = i + 1
        print(f"Zone {i+1} has {np.sum(zone_mask)} pixels")
    
    # Create visualization with dynamic colors based on number of exits
    zone_colors = [
        (255, 0, 0),    # Red
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (128, 0, 128),  # Purple
        (255, 128, 0),  # Orange
        (0, 255, 255),  # Cyan
        (128, 128, 0),  # Olive
        (255, 0, 255),  # Magenta
    ]
    
    while len(zone_colors) < num_actual_exits:
        new_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        zone_colors.append(new_color)
    
    zone_vis = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    zone_vis[binary == 0] = (0, 0, 0)  # Obstacles in black
    
    for i in range(num_actual_exits):
        mask = (zone_map == i + 1)
        if np.any(mask):
            zone_vis[mask] = zone_colors[i]
    
    zone_vis[valid_mask & ~reachable] = (128, 128, 128)
    zone_vis[exits_mask > 0] = (0, 255, 0)
    
    cv2.imwrite(os.path.join(debug_dir, '5_exit_zones.png'), zone_vis)
    
    # Convert to vector annotations
    zone_annotations = []
    for i in range(num_actual_exits):
        zone_binary = (zone_map == i + 1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(zone_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter out tiny regions
                epsilon = 0.001 * cv2.arcLength(contour, True)
                simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                segmentation = simplified_contour.flatten().tolist()
                x, y, w, h = cv2.boundingRect(simplified_contour)
                
                annotation = {
                    'id': 2000 + len(zone_annotations) + 1,
                    'category_id': 5,
                    'bbox': [x, y, w, h],
                    'segmentation': [segmentation],
                    'area': cv2.contourArea(contour),
                    'exit_id': i + 1
                }
                zone_annotations.append(annotation)
    
    return zone_annotations

def create_parallel_distance_map(binary: np.ndarray, exit_points: list, exit_point_ids: list) -> tuple:
    """Create a single distance map for all exits in parallel, treating exits as continuous lines."""
    height, width = binary.shape
    dist_map = np.full((height, width), np.inf)
    exit_map = np.full((height, width), -1)
    
    # Group points by exit ID to form line segments
    exit_segments = {}
    for (py, px), exit_id in zip(exit_points, exit_point_ids):
        if exit_id not in exit_segments:
            exit_segments[exit_id] = []
        exit_segments[exit_id].append((py, px))
    
    # Sort points in each segment to form a continuous line
    for exit_id in exit_segments:
        points = np.array(exit_segments[exit_id])
        # Sort points based on their position (this assumes exits are roughly horizontal or vertical)
        if np.ptp(points[:, 0]) > np.ptp(points[:, 1]):  # Vertical exit
            points = points[points[:, 0].argsort()]
        else:  # Horizontal exit
            points = points[points[:, 1].argsort()]
        exit_segments[exit_id] = points
    
    # Basic costs
    STRAIGHT_COST = 1.0
    DIAGONAL_COST = np.sqrt(2)
    
    # Initialize queue with points near exits
    queue = []
    for exit_id, points in exit_segments.items():
        # Create a mask for this exit
        exit_mask = np.zeros((height, width), dtype=np.uint8)
        for i in range(len(points) - 1):
            y1, x1 = points[i]
            y2, x2 = points[i + 1]
            # Draw line segment
            cv2.line(exit_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
        
        # Find points adjacent to the exit
        for y in range(height):
            for x in range(width):
                if exit_mask[y, x]:
                    # Calculate minimum distance to any point on the exit line segments
                    min_dist = np.inf
                    for i in range(len(points) - 1):
                        p1 = points[i]
                        p2 = points[i + 1]
                        # Calculate distance to line segment
                        dist = point_to_line_segment_distance(y, x, p1[0], p1[1], p2[0], p2[1])
                        min_dist = min(min_dist, dist)
                    
                    dist_map[y, x] = min_dist
                    exit_map[y, x] = exit_id
                    heapq.heappush(queue, (min_dist, y, x, exit_id))
    
    # All possible moves
    directions = [
        (0, 1, STRAIGHT_COST),   # right
        (1, 0, STRAIGHT_COST),   # down
        (0, -1, STRAIGHT_COST),  # left
        (-1, 0, STRAIGHT_COST),  # up
        (1, 1, DIAGONAL_COST),   # down-right
        (-1, -1, DIAGONAL_COST), # up-left
        (1, -1, DIAGONAL_COST),  # down-left
        (-1, 1, DIAGONAL_COST)   # up-right
    ]
    
    while queue:
        d, cy, cx, current_exit = heapq.heappop(queue)
        
        if d > dist_map[cy, cx]:
            continue
        
        # Check if near obstacle
        near_obstacle = False
        for dy, dx, _ in directions:
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < height and 0 <= nx < width and 
                binary[ny, nx] == 0):
                near_obstacle = True
                break
        
        if not near_obstacle:
            # Process neighbors normally
            for dy, dx, cost in directions:
                ny, nx = cy + dy, cx + dx
                
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                    
                if binary[ny, nx] == 0:
                    continue
                
                new_d = d + cost
                
                # Update if this is a better path
                if new_d < dist_map[ny, nx]:
                    dist_map[ny, nx] = new_d
                    exit_map[ny, nx] = current_exit
                    heapq.heappush(queue, (new_d, ny, nx, current_exit))
        else:
            # Look ahead near obstacles
            for look_dist in range(1, 21):
                for dy, dx, cost in directions:
                    ny = cy + dy * look_dist
                    nx = cx + dx * look_dist
                    
                    if not (0 <= ny < height and 0 <= nx < width):
                        continue
                        
                    if binary[ny, nx] == 0:
                        continue
                    
                    # Check if path is clear
                    path_clear = True
                    path_points = []
                    
                    for step in range(1, look_dist + 1):
                        check_y = cy + dy * step
                        check_x = cx + dx * step
                        
                        if not (0 <= check_y < height and 0 <= check_x < width):
                            path_clear = False
                            break
                            
                        if binary[check_y, check_x] == 0:
                            path_clear = False
                            break
                        
                        path_points.append((check_y, check_x))
                    
                    if not path_clear:
                        continue
                    
                    # Update all points along the path
                    for i, (py, px) in enumerate(path_points):
                        point_cost = d + cost * (i + 1)
                        
                        if point_cost < dist_map[py, px]:
                            dist_map[py, px] = point_cost
                            exit_map[py, px] = current_exit
                            heapq.heappush(queue, (point_cost, py, px, current_exit))
    
    return dist_map, exit_map

def point_to_line_segment_distance(py: float, px: float, y1: float, x1: float, y2: float, x2: float) -> float:
    """Calculate the minimum distance from a point to a line segment."""
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D

    if len_sq == 0:
        # Point 1 and 2 are the same - just return direct distance
        return np.sqrt(A * A + B * B)

    param = dot / len_sq

    if param < 0:
        # Point projects beyond first endpoint
        return np.sqrt(A * A + B * B)
    elif param > 1:
        # Point projects beyond second endpoint
        return np.sqrt((px - x2) * (px - x2) + (py - y2) * (py - y2))
    else:
        # Normal case - project point onto line
        x = x1 + param * C
        y = y1 + param * D
        return np.sqrt((px - x) * (px - x) + (py - y) * (py - y))

def calculate_path_cost(binary: np.ndarray, cy: int, cx: int, ny: int, nx: int, 
                       obstacle_distances: np.ndarray) -> float:
    """Calculate the cost of moving from (cy,cx) to (ny,nx) considering obstacles."""
    # Base cost is Euclidean distance
    base_cost = np.sqrt((ny - cy)**2 + (nx - cx)**2)
    
    # Get obstacle distances at current and next positions
    current_dist = obstacle_distances[cy, cx]
    next_dist = obstacle_distances[ny, nx]
    
    # Use the minimum distance to any obstacle along the path
    proximity = min(current_dist, next_dist)
    
    # Calculate proximity penalty (higher when closer to obstacles)
    proximity_penalty = max(0, (5 - proximity) / 5) * 0.5
    
    # Return total cost including proximity penalty
    return base_cost * (1.0 + proximity_penalty)

def visualize_floor_plan(json_path: str, debug_dir: str) -> np.ndarray:
    """Create a simple visualization of the floor plan from COCO annotations."""
    # Create debug directory if it doesn't exist
    os.makedirs(debug_dir, exist_ok=True)
    
    # Load COCO annotations
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get image dimensions from the first image
    height = data['images'][0]['height']
    width = data['images'][0]['width']
    print(f"Image dimensions: {width}x{height}")
    
    # Create blank image with white background
    visualization = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    
    # Save original annotations for later
    original_annotations = data['annotations']
    
    # Draw original annotations first
    colors = {
        1: (0, 0, 0),      # Black for Obstructions
        2: (0, 0, 0),      # Black for Variable Obstructions
        3: (0, 255, 0),    # Green for Exits
        4: (0, 0, 0)       # Black for Narrow Passages
    }
    
    # Create a mask for exits
    exits_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw original annotations and create exits mask
    for annotation in original_annotations:
        category_id = annotation['category_id']
        color = colors[category_id]
        
        if 'segmentation' in annotation and isinstance(annotation['segmentation'], list) and len(annotation['segmentation']) > 0:
            points = np.array(annotation['segmentation'][0]).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(visualization, [points], color)
            if category_id == 3:  # Exit
                cv2.fillPoly(exits_mask, [points], 255)
        else:
            bbox = annotation['bbox']
            x, y, w, h = map(int, bbox)
            cv2.rectangle(visualization, (x, y), (x + w, y + h), color, -1)
            if category_id == 3:  # Exit
                cv2.rectangle(exits_mask, (x, y), (x + w, y + h), 255, -1)
    
    # Save initial visualization
    cv2.imwrite(os.path.join(debug_dir, '1_starting_image.png'), visualization)
    
    # Get maximum door width from vector coordinates
    max_door_width = get_max_door_width(original_annotations)
    print(f"Maximum door width: {max_door_width}")
    
    # Convert to binary image (0 for obstacles, 255 for passable space)
    binary = cv2.cvtColor(visualization, cv2.COLOR_BGR2GRAY)
    # Use a lower threshold (128) to consider more pixels as passable space
    binary = cv2.threshold(binary, 128, 255, cv2.THRESH_BINARY)[1]
    
    # Save binary image
    cv2.imwrite(os.path.join(debug_dir, '2_binary.png'), binary)
    
    # Find narrow passages as vector annotations
    narrow_annotations = find_narrow_passages(binary, max_door_width, debug_dir)
    
    # Create visualization of just narrow passages
    narrow_visualization = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    for annotation in narrow_annotations:
        points = np.array(annotation['segmentation'][0]).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(narrow_visualization, [points], (0, 0, 0))
    cv2.imwrite(os.path.join(debug_dir, '3_narrow_passages.png'), narrow_visualization)
    
    # Create combined data with preserved categories
    combined_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'images': data.get('images', []),
        'categories': data['categories'] + [{
            'id': 4,
            'name': 'Narrow Passage',
            'supercategory': ''
        }],
        'annotations': original_annotations + narrow_annotations
    }
    
    # Save combined annotations
    combined_json_path = os.path.join(debug_dir, 'final.json')
    with open(combined_json_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    # Create final visualization
    final_visualization = visualization.copy()
    
    # Create a mask for all narrow passages
    narrow_mask = np.zeros((height, width), dtype=np.uint8)
    for annotation in narrow_annotations:
        points = np.array(annotation['segmentation'][0]).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(narrow_mask, [points], 255)
    
    # Remove narrow passages that overlap with exits
    narrow_mask[exits_mask > 0] = 0
    
    # Draw remaining narrow passages
    final_visualization[narrow_mask > 0] = (0, 0, 0)
    
    # Save final result
    cv2.imwrite(os.path.join(debug_dir, '4_final.png'), final_visualization)
    
    # Create exit zones
    zone_annotations = create_exit_zones(final_visualization, debug_dir)
    
    # Add zone annotations to combined data
    combined_data['categories'].append({
        'id': 5,
        'name': 'Exit Zone',
        'supercategory': ''
    })
    combined_data['annotations'].extend(zone_annotations)
    
    # Save updated combined annotations
    with open(combined_json_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    return final_visualization

if __name__ == "__main__":
    # Use absolute paths
    base_dir = r"C:\Users\BenWright\Downloads\NEW JUNK\Junk\Super Secret Projet"
    json_path = os.path.join(base_dir, "ETD Project", "Test Case - Car Parks", "Labeled", "cocotestcase", "annotations", "test.json")
    debug_dir = os.path.join(os.path.dirname(json_path), 'debug')
    
    # Create and save visualization
    visualization = visualize_floor_plan(json_path, debug_dir)
    output_path = os.path.join(os.path.dirname(json_path), 'floor_plan.png')
    cv2.imwrite(output_path, visualization)
    print(f"Floor plan visualization saved to: {output_path}") 



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

import numpy as np
import cv2
from typing import List, Tuple, Optional
import random
from skimage.draw import polygon

class RoomDropoutStrategy:
    """
    Strategy for randomly dropping rooms from a density map using ground truth coordinates.
    
    Density map: grayscale image where foreground (rooms) are white points and background is black
    GT room coordinates: list of 2D points defining each room's boundary
    """
    
    def __init__(self, density_map: np.ndarray, room_coordinates: List[List[Tuple[int, int]]]):
        """
        Initialize the dropout strategy.
        
        Args:
            density_map: Grayscale image (H, W) where white pixels represent rooms
            room_coordinates: List of rooms, each room is a list of (x, y) coordinate tuples
        """
        self.original_density_map = density_map.copy()
        self.room_coordinates = room_coordinates
        self.num_rooms = len(room_coordinates)
    
    def create_room_masks(self) -> List[np.ndarray]:
        """
        Create binary masks for each room using their GT coordinates.
        
        Returns:
            List of binary masks, one for each room
        """
        h, w = self.original_density_map.shape
        room_masks = []
        
        for room_coords in self.room_coordinates:
            mask = np.zeros((h, w), dtype=np.uint8)
            
            if len(room_coords) >= 3:  # Need at least 3 points for a polygon
                # Convert coordinates to numpy array
                coords = np.array(room_coords)
                x_coords = coords[:, 0]
                y_coords = coords[:, 1]
                
                # Create polygon mask using skimage
                rr, cc = polygon(y_coords, x_coords, shape=(h, w))
                mask[rr, cc] = 1
            
            room_masks.append(mask)
        
        return room_masks
    
    def drop_rooms_random(self, dropout_rate: float = 0.3, seed: Optional[int] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Randomly drop rooms from the density map.
        
        Args:
            dropout_rate: Fraction of rooms to drop (0.0 to 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (modified_density_map, list_of_dropped_room_indices)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Determine number of rooms to drop
        num_to_drop = int(self.num_rooms * dropout_rate)
        
        # Randomly select room indices to drop
        room_indices = list(range(self.num_rooms))
        dropped_indices = random.sample(room_indices, num_to_drop)
        
        return self._apply_dropout(dropped_indices), dropped_indices
    
    def drop_rooms_by_indices(self, room_indices: List[int]) -> np.ndarray:
        """
        Drop specific rooms by their indices.
        
        Args:
            room_indices: List of room indices to drop
            
        Returns:
            Modified density map with specified rooms removed
        """
        return self._apply_dropout(room_indices)
    
    def drop_rooms_by_area(self, min_area: Optional[int] = None, 
                          max_area: Optional[int] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Drop rooms based on their area constraints.
        
        Args:
            min_area: Minimum area threshold (drop rooms smaller than this)
            max_area: Maximum area threshold (drop rooms larger than this)
            
        Returns:
            Tuple of (modified_density_map, list_of_dropped_room_indices)
        """
        room_masks = self.create_room_masks()
        dropped_indices = []
        
        for i, mask in enumerate(room_masks):
            area = np.sum(mask)
            
            should_drop = False
            if min_area is not None and area < min_area:
                should_drop = True
            if max_area is not None and area > max_area:
                should_drop = True
                
            if should_drop:
                dropped_indices.append(i)
        
        return self._apply_dropout(dropped_indices), dropped_indices
    
    def _apply_dropout(self, room_indices_to_drop: List[int]) -> np.ndarray:
        """
        Apply dropout by removing specified rooms from the density map.
        
        Args:
            room_indices_to_drop: List of room indices to remove
            
        Returns:
            Modified density map with rooms removed
        """
        modified_map = self.original_density_map.copy()
        room_masks = self.create_room_masks()
        
        # Remove each specified room
        for room_idx in room_indices_to_drop:
            if 0 <= room_idx < len(room_masks):
                mask = room_masks[room_idx]
                # Set pixels in the room area to background (black/0)
                modified_map[mask == 1] = 0
        
        return modified_map
    
    def visualize_dropout(self, original_map: np.ndarray, modified_map: np.ndarray, 
                         dropped_indices: List[int]) -> np.ndarray:
        """
        Create a visualization showing the dropout effect.
        
        Args:
            original_map: Original density map
            modified_map: Modified density map after dropout
            dropped_indices: Indices of dropped rooms
            
        Returns:
            Visualization image with original and modified maps side by side
        """
        h, w = original_map.shape
        
        # Create side-by-side comparison
        vis = np.zeros((h, w * 2), dtype=np.uint8)
        vis[:, :w] = original_map
        vis[:, w:] = modified_map
        
        # Highlight dropped rooms in red on the original map
        if len(dropped_indices) > 0:
            room_masks = self.create_room_masks()
            vis_color = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            
            for idx in dropped_indices:
                if 0 <= idx < len(room_masks):
                    mask = room_masks[idx]
                    # Highlight in red on the left (original) side
                    vis_color[mask == 1, 0] = 0      # Blue channel
                    vis_color[mask == 1, 1] = 0      # Green channel  
                    vis_color[mask == 1, 2] = 255    # Red channel
            
            return vis_color
        
        return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

# Example usage and testing
def example_usage():
    """
    Example of how to use the RoomDropoutStrategy class.
    """
    # Create a sample density map (200x200 image)
    density_map = np.zeros((200, 200), dtype=np.uint8)
    
    # Create some sample room coordinates (rectangles and polygons)
    room_coordinates = [
        # Room 1: Rectangle
        [(20, 20), (80, 20), (80, 60), (20, 60)],
        # Room 2: Another rectangle
        [(100, 30), (180, 30), (180, 80), (100, 80)],
        # Room 3: L-shaped room
        [(30, 100), (90, 100), (90, 130), (60, 130), (60, 160), (30, 160)],
        # Room 4: Triangle
        [(120, 120), (160, 120), (140, 160)],
        # Room 5: Pentagon
        [(50, 180), (70, 170), (90, 180), (80, 195), (40, 195)]
    ]
    
    # Fill the density map with white pixels for each room
    for room_coords in room_coordinates:
        coords = np.array(room_coords)
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        from skimage.draw import polygon
        rr, cc = polygon(y_coords, x_coords, shape=density_map.shape)
        density_map[rr, cc] = 255  # White pixels for rooms
    
    # Initialize the dropout strategy
    dropout_strategy = RoomDropoutStrategy(density_map, room_coordinates)
    
    # Example 1: Random dropout
    print("Example 1: Random dropout (30% of rooms)")
    modified_map1, dropped_indices1 = dropout_strategy.drop_rooms_random(dropout_rate=0.3, seed=42)
    print(f"Dropped rooms: {dropped_indices1}")
    
    # Example 2: Drop specific rooms
    print("\nExample 2: Drop specific rooms (indices 0 and 2)")
    modified_map2 = dropout_strategy.drop_rooms_by_indices([0, 2])
    
    # Example 3: Drop rooms by area
    print("\nExample 3: Drop rooms with area > 3000 pixels")
    modified_map3, dropped_indices3 = dropout_strategy.drop_rooms_by_area(max_area=3000)
    print(f"Dropped rooms by area: {dropped_indices3}")
    
    return density_map, modified_map1, modified_map2, modified_map3

if __name__ == "__main__":
    example_usage()
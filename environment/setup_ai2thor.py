
from ai2thor.controller import Controller
import matplotlib.pyplot as plt
import random

class ThorEnvironment:
    """
    A wrapper for the AI2-THOR environment to simplify interaction.
    """
    def __init__(self, scene_name='FloorPlan1'):
        print("Initializing AI2-THOR Controller...")
        self.controller = Controller(
            scene=scene_name,
            gridSize=0.25,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=400,
            height=300
        )
        print("Controller initialized.")

    def reset(self, scene_name=None):
        """Resets the environment to a new scene or the current one."""
        if scene_name:
            self.controller.reset(scene=scene_name)
        else:
            self.controller.reset()
            
        # Teleport to a random reachable position
        event = self.controller.step(action='GetReachablePositions')
        random_pos = random.choice(event.metadata['actionReturn'])
        random_rotation = random.choice([0, 90, 180, 270])
        
        event = self.controller.step(
            action='Teleport',
            position=random_pos,
            rotation=dict(x=0, y=random_rotation, z=0),
            horizon=0
        )
        return event.frame

    def step(self, action):
        """
        Takes an action and returns the new observation.
        
        Args:
            action (str): One of 'MoveAhead', 'RotateRight', 'RotateLeft'.
            
        Returns:
            np.ndarray: The new image frame from the agent's perspective.
        """
        event = self.controller.step(action=action)
        return event.frame
    
    def get_current_frame(self):
        """Returns the current camera frame."""
        return self.controller.last_event.frame

if __name__ == '__main__':
    # --- Example of how to use the environment ---
    env = ThorEnvironment(scene_name='FloorPlan2_physics')
    
    print("\nResetting environment and getting initial frame...")
    initial_frame = env.reset()
    
    print("Taking a few random actions...")
    frame_after_move = env.step('MoveAhead')
    frame_after_turn = env.step('RotateRight')
    
    # Display the frames to verify
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(initial_frame)
    axes[0].set_title("Initial View")
    axes[0].axis('off')
    
    axes[1].imshow(frame_after_move)
    axes[1].set_title("After Moving Ahead")
    axes[1].axis('off')
    
    axes[2].imshow(frame_after_turn)
    axes[2].set_title("After Turning Right")
    axes[2].axis('off')
    
    plt.show()
    
    print("Environment interaction verified.")

import cv2
import numpy as np
import base64
import io
import json
import time
import queue
import threading
from PIL import Image
from IPython.display import display, Javascript, HTML
from google.colab.output import eval_js
from base64 import b64decode, b64encode

class CrapsGameController:
    def __init__(self):
        self.event_queue = queue.Queue()
        self.frame_buffer = ThreadSafeFrameBuffer()
        self.webcam_buffer = ThreadSafeFrameBuffer()  # New buffer for webcam frames
        self.game_thread = None
        self.game_running = False
        self.current_mode = 'manual'
        self.webcam_active = False
        
        # Statistics
        self.stats = {
            'manual': 0,
            'cnn': 0,
            'yolo': 0
        }
        
        # Performance tracking
        self.fps = 0
        self.fps_update_time = time.time()
        self.fps_frame_count = 0
        self.last_frame_number = 0
        
        # Webcam tracking
        self.last_webcam_time = time.time()
        self.webcam_frame_count = 0
        
    def start_game_thread(self):
        """Start the game thread"""
        if not self.game_running:
            self.game_running = True
            
            # Send initial mode to game thread
            self.event_queue.put(f"mode:{self.current_mode}")
            
            # Start the game thread with webcam buffer
            self.game_thread = threading.Thread(
                target=game_with_frame_buffer,
                args=(self.event_queue, self.frame_buffer, self.webcam_buffer),
                daemon=True
            )
            self.game_thread.start()
            print("🎰 Game thread started!")
            
    def stop_game_thread(self):
        """Stop the game thread"""
        if self.game_running:
            self.event_queue.put("quit")
            if self.game_thread:
                self.game_thread.join(timeout=2.0)
            self.game_running = False
            print("🚪 Game thread stopped.")
            
    def handle_event(self, event_type, event_data=None):
        """Handle events from JavaScript"""
        if event_type == "start":
            self.start_game_thread()
            
        elif event_type == "quit":
            self.stop_game_thread()
            
        elif event_type == "roll":
            if self.current_mode == 'manual':
                self.event_queue.put("roll")
                self.stats['manual'] += 1
                
        elif event_type == "mode":
            self.current_mode = event_data
            if self.game_running:
                self.event_queue.put(f"mode:{self.current_mode}")
                
    def process_webcam_frame(self, webcam_data):
        """Process webcam frame from JavaScript and send to game thread"""
        if webcam_data and webcam_data.get('frame'):
            try:
                # Decode base64 image
                image_data = webcam_data['frame'].split(',')[1]
                image_bytes = b64decode(image_data)
                
                # Convert to numpy array
                jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
                frame = cv2.imdecode(jpg_as_np, flags=cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Write to webcam buffer for game thread to access
                    self.webcam_buffer.write(frame)
                    
                    # Track webcam FPS
                    self.webcam_frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_webcam_time >= 1.0:
                        webcam_fps = self.webcam_frame_count / (current_time - self.last_webcam_time)
                        self.webcam_frame_count = 0
                        self.last_webcam_time = current_time
                        # print(f"📷 Webcam FPS: {webcam_fps:.1f}")
                    
                    return True
            except Exception as e:
                print(f"Error processing webcam frame: {e}")
                return False
        return False
                
    def get_current_frame(self):
        """Get current frame and performance metrics"""
        frame, frame_number = self.frame_buffer.read()
        
        if frame is not None and frame_number > self.last_frame_number:
            self.last_frame_number = frame_number
            
            # Update FPS counter
            self.fps_frame_count += 1
            current_time = time.time()
            if current_time - self.fps_update_time >= 1.0:
                self.fps = self.fps_frame_count / (current_time - self.fps_update_time)
                self.fps_frame_count = 0
                self.fps_update_time = current_time
            
            # Convert frame to base64
            frame_base64 = image_to_base64(frame)
            
            return {
                'frame': frame_base64,
                'fps': self.fps,
                'frame_number': frame_number,
                'has_update': True
            }
        
        return {
            'has_update': False
        }


class ThreadSafeFrameBuffer:
    """Thread-safe frame buffer for passing frames between threads"""
    def __init__(self):
        self.frame = None
        self.frame_number = 0
        self.lock = threading.Lock()
        
    def write(self, frame):
        """Write a new frame (from game thread or webcam)"""
        with self.lock:
            # Convert to numpy array if needed
            if hasattr(frame, 'get_buffer'):
                # It's a pygame surface
                w, h = frame.get_size()
                # Use pygame.surfarray for faster conversion
                import pygame as pg
                frame_array = pg.surfarray.array3d(frame)
                # Transpose to get correct orientation
                self.frame = np.transpose(frame_array, (1, 0, 2))
            else:
                # Already numpy array
                self.frame = np.array(frame)
            
            self.frame_number += 1
    
    def read(self):
        """Read the current frame (from display thread)"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.frame_number
            return None, 0


def image_to_base64(img, format='png'):
    """Convert numpy array to base64 string"""
    if img is None:
        return ""
    
    # Ensure RGB format
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # If already RGB, no conversion needed
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img.astype('uint8'))
    
    # Save to bytes
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format.upper(), optimize=False)
    
    # Encode to base64
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format};base64,{img_str}"


def create_craps_interface():
    """Display the HTML interface"""
    display(HTML(filename="craps_game.html"))


def run_craps_game():
    """Main function to run the craps game system"""
    
    # Create the interface
    create_craps_interface()
    
    # Initialize controller
    controller = CrapsGameController()
    
    print("🎰 Craps Game Controller Ready!")
    print("1. Click 'Start Game' to begin")
    print("2. Select your game mode (Manual or AI)")
    print("3. In AI mode, click 'Start Webcam' to enable dice detection")
    print("4. Click 'Roll Dice' in manual mode or let AI play")
    print("5. Click 'Cash Out' to stop the game")
    
    # Main processing loop
    while True:
        try:
            # Get state from JavaScript
            js_state = eval_js('window.getGameState()')
            
            if not js_state:
                time.sleep(0.05)
                continue
            
            # Handle state changes
            if js_state.get('running') and not controller.game_running:
                controller.handle_event("start")
                
            elif not js_state.get('running') and controller.game_running:
                controller.handle_event("quit")
                break
            
            # Handle mode changes
            if js_state.get('mode') != controller.current_mode:
                controller.handle_event("mode", js_state.get('mode'))
            
            # Handle webcam state
            controller.webcam_active = js_state.get('webcamActive', False)
            
            # Check for manual roll (stats change indicates roll)
            if js_state.get('stats', {}).get('manual', 0) > controller.stats['manual']:
                controller.handle_event("roll")
            
            # Get webcam frame if in AI mode and webcam is active
            if controller.webcam_active and controller.current_mode in ['cnn', 'yolo']:
                webcam_data = eval_js('window.getWebcamFrame()')
                if webcam_data:
                    controller.process_webcam_frame(webcam_data)
            
            # Get and update game frame
            frame_data = controller.get_current_frame()
            
            if frame_data['has_update']:
                # Update the display
                eval_js(f'''
                    window.updateGameFrame(
                        "{frame_data['frame']}", 
                        {frame_data['fps']}, 
                        {frame_data['frame_number']}
                    )
                ''')
            
            # Small delay to prevent CPU overuse
            time.sleep(0.033)  # ~30 FPS
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping craps game...")
            controller.stop_game_thread()
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("🎰 Craps game stopped.")


# Modified game function that uses frame buffer and webcam
def game_with_frame_buffer(event_queue, frame_buffer, webcam_buffer):
    """Game thread that sends frames to buffer and receives webcam frames"""
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    import pygame as pg
    from craps.craps import Craps 
    from craps.casino_player import CasinoPlayer
    from craps import prepare
    
    def create_new_games_stats():
        stats = dict()
        stats['cash'] = prepare.MONEY
        stats['account balance'] = 0
        stats['Craps'] = {
            'times as shooter': 0,
            'bets placed': 0,
            'bets won': 0,
            'bets lost': 0,
            'total bets': 0,
            'total winnings': 0
        }
        return stats
    
    # Initialize pygame
    pg.init()
    
    # Set up the screen and clock
    screen = pg.display.set_mode(prepare.RENDER_SIZE, pg.RESIZABLE)
    pg.display.set_caption("Craps Game")
    clock = pg.time.Clock()
    
    # Create player stats and initialize
    stats = create_new_games_stats()
    casino_player = CasinoPlayer(stats)
    
    # Initialize the Craps game
    game = Craps()
    game.startup(pg.time.get_ticks(), {"casino_player": casino_player})
    
    # Game loop variables
    running = True
    dt = 0
    scale = (1, 1)
    current_mode = 'manual'
    
    # AI detection variables
    last_detection_time = time.time()
    detection_cooldown = 5.0  # seconds between detections
    
    print("Game thread: Starting game loop...")
    
    while running:
        # Process events from queue
        try:
            event = event_queue.get(block=False)
            
            if event == "quit":
                print("Game thread: Quitting the game...")
                running = False
                
            elif event == "roll":
                # Manual roll
                print("Game thread: Manual roll triggered")
                game.roll(mode="manual")
                
            elif event.startswith("mode:"):
                # Update mode
                current_mode = event.split(":")[1]
                print(f"Game thread: Mode set to {current_mode}")
                
            event_queue.task_done()
        except queue.Empty:
            pass
        
        # AI mode processing - get webcam frame and detect dice
        if current_mode in ["cnn", "yolo"]:
            # print(f"Game thread: reading webcam_buffer")
            webcam_frame, webcam_frame_num = webcam_buffer.read()
            if webcam_frame is not None:
                # Check cooldown to avoid too frequent detections
                current_time = time.time()
                if current_time - last_detection_time >= detection_cooldown:
                    last_detection_time = current_time
                    game.roll(mode="cnn", frame=webcam_frame)

                    # TODO: Implement actual dice detection here
                    # For now, we'll just simulate detection
                    # You would call your CNN or YOLO model here
                    
                    # Example placeholder for dice detection:
                    # detected_dice = detect_dice_with_model(webcam_frame, current_mode)
                    # if detected_dice:
                    #     print(f"Game thread: Detected dice values: {detected_dice}")
                    #     game.roll(current_mode, dice_values=detected_dice)
                    
                    # For demonstration, just print that we have a webcam frame
                    # print(f"Game thread: Processing webcam frame in {current_mode} mode (frame #{webcam_frame_num})")
                    # print(f"  Frame shape: {webcam_frame.shape}")
                    
                    # You can add your detection logic here:
                    # 1. Preprocess the webcam frame
                    # 2. Run through your CNN or YOLO model
                    # 3. Extract dice values
                    # 4. Trigger game roll with detected values
        
        # Update game state
        keys = pg.key.get_pressed()
        current_time = pg.time.get_ticks()
        game.update(screen, keys, current_time, dt, scale)
        
        # Send frame to buffer
        frame_buffer.write(screen)
        
        # Cap frame rate
        dt = clock.tick(30)  # 30 FPS
    
    print("Game Thread: Closing...")
    pg.quit()
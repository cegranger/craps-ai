import ipywidgets as widgets
from IPython.display import display
import queue
import threading
import time
from datetime import datetime
import numpy as np
from PIL import Image
import io

class ThreadSafeFrameBuffer:
    """Thread-safe frame buffer for passing frames between threads"""
    def __init__(self):
        self.frame = None
        self.frame_number = 0
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        
    def write(self, frame):
        """Write a new frame (from game thread)"""
        with self.lock:
            # Convert pygame surface to numpy array if needed
            if hasattr(frame, 'get_buffer'):
                # It's a pygame surface
                w, h = frame.get_size()
                buf = frame.get_buffer()
                # Convert to numpy array (RGB format)
                self.frame = np.frombuffer(buf.raw, dtype=np.uint8).reshape((h, w, 3))
            else:
                # Already numpy array or PIL image
                self.frame = np.array(frame)
            
            self.frame_number += 1
            self.new_frame_event.set()
    
    def read(self):
        """Read the current frame (from display thread)"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.frame_number
            return None, 0
    
    def wait_for_frame(self, timeout=0.1):
        """Wait for a new frame with timeout"""
        return self.new_frame_event.wait(timeout)
    
    def clear_event(self):
        """Clear the new frame event"""
        self.new_frame_event.clear()

class CrapsGameController:
    def __init__(self, event_queue, frame_buffer, **kwargs):
        self.event_queue = event_queue
        self.frame_buffer = frame_buffer
        self.event_count = {'roll': 0, 'quit': 0}
        self.game_running = False
        self.last_frame_number = 0
        self.fps = 0
        self.fps_update_time = time.time()
        self.fps_frame_count = 0
        self.use_yolo = kwargs.get("use_yolo", False)
        self.model = kwargs.get("model", None)
        
        # Create widgets
        self.create_widgets()
        
    def create_widgets(self):
        # Frame display widget
        self.frame_display = widgets.Image(
            format='png',
            width=800,
            height=600,
            layout=widgets.Layout(
                border='3px solid #333',
                border_radius='5px'
            )
        )

        # Game control buttons
        if self.use_yolo:
            self.roll_button = None
        else:
            self.roll_button = widgets.Button(
                description='🎲 Roll Dice',
                button_style='success',
                tooltip='Roll the dice!',
                layout=widgets.Layout(width='150px', height='50px')
            )
        
        self.quit_button = widgets.Button(
            description='🚪 Cash Out',
            button_style='danger',
            tooltip='Quit the game',
            layout=widgets.Layout(width='150px', height='50px')
        )
        
        self.start_game_button = widgets.Button(
            description='🎰 Start Game',
            button_style='primary',
            tooltip='Start the game thread',
            layout=widgets.Layout(width='150px', height='50px')
        )
        
        # Performance info
        self.performance_label = widgets.HTML(
            value=self._get_performance_html()
        )
        
        # Game status
        self.game_status = widgets.HTML(
            value=self._get_status_html()
        )
        
        # Event counters
        self.counter_label = widgets.HTML(
            value=self._get_counter_html()
        )
        
        # Output area
        self.output = widgets.Output(
            layout=widgets.Layout(
                height='200px',
                width='100%',
                border='1px solid #ddd',
                overflow_y='auto',
                padding='5px'
            )
        )
        
        # Connect handlers
        if self.roll_button:
            self.roll_button.on_click(self.on_roll_click)
        self.quit_button.on_click(self.on_quit_click)
        self.start_game_button.on_click(self.on_start_game_click)
        
        # Start frame update thread
        self.start_frame_updater()
        
    def _get_performance_html(self):
        return f"""
        <div style="font-family: monospace; padding: 5px; background: #e8f5e9; border-radius: 5px;">
            <b>⚡ Performance:</b> FPS: <span style="color: green; font-size: 1.2em;">{self.fps:.1f}</span> | 
            Frame: <span style="color: blue;">{self.last_frame_number}</span>
        </div>
        """
        
    def _get_status_html(self):
        status_color = "green" if self.game_running else "red"
        status_text = "🟢 Running" if self.game_running else "🔴 Stopped"
        return f"""
        <div style="font-family: monospace; padding: 10px; background: #f0f0f0; border-radius: 5px;">
            <h4 style="margin: 0;">Game Status: <span style="color: {status_color};">{status_text}</span></h4>
        </div>
        """
        
    def _get_counter_html(self):
        return f"""
        <div style="font-family: monospace; padding: 10px; background: #f9f9f9; border-radius: 5px;">
            <b>📊 Event Statistics:</b><br>
            🎲 Rolls: <span style="color: green; font-size: 1.2em;">{self.event_count['roll']}</span> | 
            🚪 Quits: <span style="color: red; font-size: 1.2em;">{self.event_count['quit']}</span>
        </div>
        """
        
    def start_frame_updater(self):
        """Start thread that updates frame display from shared buffer"""
        def update_loop():
            while True:
                # Wait for new frame or timeout
                if self.frame_buffer.wait_for_frame(timeout=0.05):  # 50ms timeout
                    self.frame_buffer.clear_event()
                    
                    frame, frame_number = self.frame_buffer.read()
                    if frame is not None and frame_number > self.last_frame_number:
                        # Convert numpy array to PNG
                        img = Image.fromarray(frame)
                        buffer = io.BytesIO()
                        img.save(buffer, format='PNG', optimize=False)  # No optimization for speed
                        self.frame_display.value = buffer.getvalue()
                        self.last_frame_number = frame_number
                        
                        # Update FPS counter
                        self.fps_frame_count += 1
                        current_time = time.time()
                        if current_time - self.fps_update_time >= 1.0:
                            self.fps = self.fps_frame_count / (current_time - self.fps_update_time)
                            self.fps_frame_count = 0
                            self.fps_update_time = current_time
                            self.performance_label.value = self._get_performance_html()
                
        updater_thread = threading.Thread(target=update_loop, daemon=True)
        updater_thread.start()
        
    def on_start_game_click(self, b):
        """Start the game thread"""
        if not self.game_running:
            # Start the modified game thread
            game_thread = threading.Thread(
                target=game_with_frame_buffer,
                args=(self.event_queue, self.frame_buffer, self.use_yolo, self.model),
                daemon=True
            )
            game_thread.start()
            
            self.game_running = True
            self.game_status.value = self._get_status_html()
            self.start_game_button.disabled = True
            
            with self.output:
                print(f"🎰 [{datetime.now().strftime('%H:%M:%S')}] Game thread started!")
                print("🎲 Game is ready. Click 'Roll Dice' to play!")
                
    def on_roll_click(self, b):
        self.event_queue.put("roll")
        self.event_count['roll'] += 1
        self.counter_label.value = self._get_counter_html()
        with self.output:
            print(f"🎲 [{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Rolling dice...")
            
    def on_quit_click(self, b):
        self.event_queue.put("quit")
        self.event_count['quit'] += 1
        self.counter_label.value = self._get_counter_html()
        self.game_running = False
        self.game_status.value = self._get_status_html()
        self.start_game_button.disabled = False
        with self.output:
            print(f"🚪 [{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Cashing out... Game stopping.")
            
    def display(self):
        # Title
        title = widgets.HTML('''
        <div style="text-align: center; padding: 10px; background: linear-gradient(90deg, #2e7d32, #1976d2); color: white; border-radius: 10px;">
            <h2 style="margin: 0;">🎰 Craps Game Controller 🎲</h2>
        </div>
        ''')
        
        # Game controls
        game_controls_title = widgets.HTML('<h3>🎮 Game Controls</h3>')
        buttons = [self.start_game_button, self.quit_button]
        if self.roll_button:
            buttons.insert(1, self.roll_button)
        game_buttons = widgets.HBox(
            buttons,
            layout=widgets.Layout(justify_content='space-around', margin='10px')
        )
        
        # Left panel
        left_panel = widgets.VBox([
            game_controls_title,
            game_buttons,
            self.game_status,
            self.counter_label,
            self.performance_label,
            widgets.HTML('<h4>📋 Game Log:</h4>'),
            self.output
        ], layout=widgets.Layout(width='400px', padding='10px'))
        
        # Right panel
        right_panel = widgets.VBox([
            widgets.HTML('<h3 style="text-align: center;">🎮 Game Display</h3>'),
            self.frame_display
        ], layout=widgets.Layout(padding='10px'))
        
        # Main layout
        main_content = widgets.HBox([left_panel, right_panel], layout=widgets.Layout(gap='20px'))
        
        return widgets.VBox([title, main_content])

# Modified game function that uses frame buffer instead of saving files
def game_with_frame_buffer(event_queue, frame_buffer, use_yolo, model=None):
    """Modified game thread that sends frames to buffer instead of saving files"""
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
    game = Craps(use_yolo=use_yolo, model=model)
    game.startup(pg.time.get_ticks(), {"casino_player": casino_player})
    
    # Game loop variables
    running = True
    dt = 0
    scale = (1, 1)
    
    while running:
        try:
            event = event_queue.get(block=False)
            if event == "quit":
                print("Quitting the game...")
                running = False
            elif event == "roll":
                game.roll()
            event_queue.task_done()
        except queue.Empty:
            pass
        
        # Update game state
        keys = pg.key.get_pressed()
        current_time = pg.time.get_ticks()
        game.update(screen, keys, current_time, dt, scale)
        
        # Send frame to buffer instead of saving to file
        # Convert surface to RGB array
        w, h = screen.get_size()
        # pygame.surfarray is faster but requires numpy
        frame_array = pg.surfarray.array3d(screen)
        # Transpose to get correct orientation (pygame uses (width, height, channels))
        frame_array = np.transpose(frame_array, (1, 0, 2))
        
        # Write to shared buffer
        frame_buffer.write(frame_array)
        
        # Cap frame rate
        dt = clock.tick(15)
    
    pg.quit()
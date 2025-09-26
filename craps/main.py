import pygame as pg
from craps.craps import Craps  # Import the Craps class from craps.py
from craps.casino_player import CasinoPlayer  # Import CasinoPlayer
from . import prepare

from IPython.display import display, clear_output, Image
import os
from PIL import Image as PILImage

def create_new_games_stats():
    """Create a new stats dictionary for initializing CasinoPlayer."""
    stats = dict()
    stats['cash'] = prepare.MONEY  # Default starting cash
    stats['account balance'] = 0  # Default account balance
    stats['Craps'] = {
        'times as shooter': 0,
        'bets placed': 0,
        'bets won': 0,
        'bets lost': 0,
        'total bets': 0,
        'total winnings': 0
    }
    # Add more games here if needed
    return stats

def main():
    # Initialize pygame
    pg.init()

    # Set up the screen and clock
    screen = pg.display.set_mode(prepare.RENDER_SIZE, pg.RESIZABLE)
    pg.display.set_caption("Craps Game")
    clock = pg.time.Clock()

    # Create player stats and initialize CasinoPlayer
    stats = create_new_games_stats()
    casino_player = CasinoPlayer(stats)  # Create CasinoPlayer instance

    # Initialize the Craps game
    game = Craps()
    game.startup(pg.time.get_ticks(), {"casino_player": casino_player})  # Pass CasinoPlayer to the game

    # Game loop variables
    running = True
    dt = 0
    scale = (1, 1)  # Scale for resizing (if used)
    save_frame = False  # Toggle for saving frames
    
    while running:
        # Handle events
        for event in pg.event.get():
            if event.type == pg.QUIT:  # Quit the main loop
                running = False
            elif event.type == pg.VIDEORESIZE:  # Handle window resizing
                prepare.RENDER_SIZE = event.size
                screen = pg.display.set_mode(prepare.RENDER_SIZE, pg.RESIZABLE)
                scale = (event.w / prepare.RENDER_SIZE[0], event.h / prepare.RENDER_SIZE[1])
            elif event.type == pg.KEYDOWN:  # Handle key presses
                if event.key == pg.K_f:  # Toggle frame saving with 'F' key
                    save_frame = not save_frame
                    print(f"Frame saving toggled {'ON' if save_frame else 'OFF'}")

            # Pass events to the game
            game.get_event(event, scale)

        # Update game state
        keys = pg.key.get_pressed()
        current_time = pg.time.get_ticks()
        game.update(screen, keys, current_time, dt, scale)

        # Save the frame if toggled
        if save_frame:
            pg.image.save(screen, "frame.png")
            # print("Frame saved as 'frame.png'")

        # Draw the current frame
        pg.display.flip()

        # Cap the frame rate and calculate delta time
        dt = clock.tick(60)  # Limit to 60 FPS

    # Quit pygame
    pg.quit()

if __name__ == "__main__":
    main()
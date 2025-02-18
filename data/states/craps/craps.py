import os
import queue
import random
import threading
from collections import OrderedDict

import cv2
import pygame as pg

import data.state
import models.yolo as yolo
from data import prepare, tools
from data.components.labels import ButtonGroup, Label, NeonButton, TextBox
from data.components.warning_window import InfoWindow
from data.states.craps.opencv_dice import take_picture

from . import craps_data, dice, point_chip


class Craps(data.state.State):
    show_in_lobby = True
    name = 'craps'

    def __init__(self):
        super(Craps, self).__init__()
        self.screen_rect = pg.Rect((0, 0), prepare.RENDER_SIZE)
        self.font = prepare.FONTS["Saniretro"]
        self.font_size = 64
        self.buttons = self.make_buttons(self.screen_rect)
        self.table_orig = prepare.GFX['craps_table']
        self.table_color = (0, 153, 51)
        self.set_table()
        self.bets = craps_data.BETS

        self.dice = [dice.Die(self.screen_rect), dice.Die(self.screen_rect, 50)]
        self.dice_total = 0
        self.update_total_label()
        self.history = [] #[(1,1),(5,4)]
        self.dice_sounds = [
            prepare.SFX['dice_sound1'],
            prepare.SFX['dice_sound2'],
            prepare.SFX['dice_sound3'],
            prepare.SFX['dice_sound4'],
        ]

        self.pointchip = point_chip.PointChip()
        self.points = [4,5,6,8,9,10]
        self.point = 0 #off position

        self.widgets = []
        if prepare.DEBUG:
            self.setup_debug_entry()
            self.debug_die1 = None
            self.debug_die2 = None
            self.debug_dice_total = None

        self.popup = None

        self.cap = None
        self.yolo_thread = None

        self.model = yolo.load_model(os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..",
            "models",
            "craps-ai",
            # "yolov8n",
            # "yolov8n4",
            "yolov8n-obb",
            "weights",
            "best.pt"
        ))

    @staticmethod
    def initialize_stats():
        """Return OrderedDict suitable for use in game stats

        :return: collections.OrderedDict
        """
        stats = OrderedDict([('times as shooter', 0),
                             ('bets placed', 0),
                             ('bets won', 0),
                             ('bets lost', 0),
                             ('total bets', 0),
                             ('total winnings', 0)])
        return stats

    def setup_debug_entry(self):
        self.debug_lbl = Label(self.font, self.font_size, '6 6', "gold3", {"center": (750, 950)})
        settings = {
            "command" : self.debug_roll,
            "inactive_on_enter" : False,
            'active': False
        }
        self.widgets.append(TextBox((700,1000,150,30), **settings))

    def make_buttons(self, screen_rect):
        buttons = ButtonGroup()
        y = screen_rect.bottom-NeonButton.height-10
        lobby = NeonButton((20,y), "Lobby", self.back_to_lobby, None, buttons)
        # NeonButton((lobby.rect.right+20,y), "Roll", self.roll, None, buttons)
        return buttons

    def back_to_lobby(self, *args):
        self.game_started = False
        self.next = "lobby"
        self.done = True

        # Release video capture and join thread
        self.cap.release()
        self.cap = None

        self.yolo_thread.join()
        self.yolo_thread = None

    def debug_roll(self, id, text):
        self.roll()
        try:
            die1 = int(text.split()[0]) - 1
            die2 = int(text.split()[1]) - 1
            accepted = range(0,6)
            if die1 in accepted and die2 in accepted:
                self.dice[0].roll_value = die1
                self.dice[1].roll_value = die2
            else:
                print('Input needs to be of values 1-6')
        except IndexError: #user didnt input correct format "VALUE VALUE"
            print('Input needs to be "VALUE VALUE"')

    def reset_game(self):
        print("Resetting game")
        self.history = []
        for die in self.dice:
            die.roll_value = 0
            die.draw_dice = False

        self.dice_total = 0
        self.update_total_label()

        self.point = 0

    def roll(self, **kwargs):
        if not self.dice[0].rolling:
            self.update_history()
            random.choice(self.dice_sounds).play()

            dice_values = kwargs.get("dice_values", None)
            crops = kwargs.get("crops", None)

            if not dice_values or not crops:
                self.cap.read()
                # dice_value, crops = take_picture(self.cap)
                dice_values, crops = yolo.take_picture(self.cap, self.model)

            print(f'Dice Count: {len(dice_values)}')
            if len(dice_values) == len(self.dice):
                for i, die in enumerate(self.dice):
                    print(f'Dice {i+1}: {dice_values[i]}')
                    die.reset(dice_values[i], crops[i])
                if prepare.DEBUG:
                    print(self.history)

                self.get_dice_total(pg.time.get_ticks())

                # Reset popup
                self.popup = None
                message = None

                if self.point == 0:  # Come-Out Roll (first phase)
                    if self.dice_total in (7, 11):  # Win immediately
                        print("You rolled a 7 or 11! You win!")
                        message = "You won!"

                    elif self.dice_total in (2, 3, 12):  # Craps: Lose immediately
                        print("Craps! You rolled a 2, 3, or 12. You lose!")
                        message = "You lost!"

                    else:  # Establish the point
                        self.point = self.dice_total
                        print(f"Point established at {self.point}. Roll again!")

                else:  # Point Phase (second phase)
                    if self.dice_total == self.point:  # Win by hitting the point
                        print(f"You rolled the point {self.point}! You win!")
                        message = "You won!"
                    
                    elif self.dice_total == 7:  # Lose by rolling a 7
                        print("You rolled a 7. You lose!")
                        message = "You lost!"
                    
                    else:  # Any other number, keep rolling
                        print(f"You rolled {self.dice_total}. Keep rolling!")

                if message: # Show popup
                    self.popup = InfoWindow(
                        self.screen_rect.center,
                        message,
                        self.reset_game
                    )
            else:
                print('Wrong number of dice, please re-roll')

    def set_table(self):
        self.table_y = (self.screen_rect.height // 4)*3
        self.table_x = self.screen_rect.width
        self.table = pg.transform.scale(self.table_orig, (self.table_x, self.table_y))
        self.table_rect = self.table.get_rect()

    def startup(self, current_time, persistent):
        self.persist = persistent
        #This is the object that represents the user.
        self.casino_player = self.persist["casino_player"]
        self.casino_player.current_game = self.name
        for die in self.dice:
            die.draw_dice = False
        self.history = []

        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        if self.yolo_thread and self.yolo_thread.is_alive():
            self.yolo_thread.join()
            self.yolo_thread = None

        # Video capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # QueueDict instances to store dice detection results
        self.previous_detections = yolo.QueueDict(maxsize=0)
        self.current_detections = yolo.QueueDict(maxsize=len(self.dice))

        confidence = 0.8
        stability_frames = 10
        position_frames = 1
        translate_margin = 0.75
        debug = True
        self.yolo_thread = threading.Thread(
            target=yolo.stable_predict,
            args=(
                (self.previous_detections, self.current_detections),
                self.cap,
                self.model,
                confidence,
                stability_frames,
                position_frames,
                translate_margin,
                debug
            ),
            daemon=True
        )
        self.yolo_thread.start()

    def get_event(self, event, scale=(1,1)):
        if event.type == pg.QUIT:
            #self.cash_out_player()
            self.done = True
            self.next = "lobby"
        elif event.type == pg.VIDEORESIZE:
            self.set_table()

        # If a popup is active, pass events to it instead of buttons
        if self.popup and not self.popup.done:
            self.popup.get_event(event)
        else:
            self.buttons.get_event(event)

        for widget in self.widgets:
            widget.get_event(event, tools.scaled_mouse_pos(scale))

    def cash_out_player(self):
        self.casino_player.stats["cash"] = self.player.get_chip_total()

    def update_total_label(self):
        self.dice_total_label = Label(self.font, self.font_size, str(self.dice_total), "gold3", {"center": (1165, 245)})

    def update_history(self):
        dice = []
        for die in self.dice:
            dice.append(die.value())
        if dice[0]:
            self.history.append(dice)
        if len(self.history) > 10:
            self.history.pop(0)

    def set_point(self):
        if not self.point:
            if self.dice_total in self.points:
                self.point = self.dice_total
        if self.dice_total == 7:
            self.point = 0

    def get_dice_total(self, current_time):
        self.dice_total = 0
        for die in self.dice:
            die.update(current_time)
            v = die.value()
            if v:
                self.dice_total += v

    def draw(self, surface):
        surface.fill(self.table_color)
        surface.blit(self.table, self.table_rect)
        self.buttons.draw(surface)
        for h in self.bets.keys():
            self.bets[h].draw(surface)

        for die in self.dice:
            die.draw(surface)

        if not self.dice[0].rolling and self.dice[0].draw_dice:
            self.dice_total_label.draw(surface)

            # Draw popup if active
            if self.popup and not self.popup.done:
                self.popup.draw(surface)

        self.pointchip.draw(surface)
        for widget in self.widgets:
            widget.draw(surface)
        if prepare.DEBUG:
            self.debug_lbl.draw(surface)

    def update(self, surface, keys, current_time, dt, scale):
        mouse_pos = tools.scaled_mouse_pos(scale)

        # If a popup is active, pass mouse position to it instead of buttons
        if self.popup and not self.popup.done:
            self.popup.update(mouse_pos)
        else:
            self.buttons.update(mouse_pos)
            for h in self.bets.keys():
                self.bets[h].update(mouse_pos, self.point)

        self.draw(surface)
        self.get_dice_total(current_time)

        if self.popup and not self.popup.done:
            self.point = 0
            self.pointchip.update(current_time, 7, self.dice[0])
        else:
            self.set_point()
            self.pointchip.update(current_time, self.dice_total, self.dice[0])

        self.update_total_label()
        for widget in self.widgets:
            widget.update()

        # print("Previous:", self.previous_detections.keys())
        # print("Current:", self.current_detections.keys())
        if len(self.current_detections) == len(self.dice):
            has_new_dice = set(self.current_detections.keys()) - set(self.previous_detections.keys())
            if has_new_dice:
                dice_values = []
                crops = []
                for key, value in self.current_detections.items():
                    if self.previous_detections.full():
                        self.previous_detections.pop()
                    self.previous_detections[key] = value

                    dice_values.append(int(value[0]))
                    crops.append(value[1])

                self.roll(dice_values=dice_values, crops=crops)

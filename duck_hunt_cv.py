"""
╔══════════════════════════════════════════════════════════════════╗
║          DUCK HUNT CV - Hand Gesture Controlled Game             ║
║          Uses OpenCV + MediaPipe + Pygame                        ║
╠══════════════════════════════════════════════════════════════════╣
║  INSTALL DEPENDENCIES:                                           ║
║    pip install opencv-python mediapipe pygame numpy              ║
║                                                                  ║
║  CONTROLS:                                                       ║
║    • Point with index finger to aim                              ║
║    • Pinch thumb + index finger to SHOOT                         ║
║    • Press 'Q' to quit, 'P' to pause                            ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import time
import math
import random
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

# ─────────────────────────────────────────────
#  CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────
WINDOW_TITLE = "Duck Hunt CV"
TARGET_FPS    = 30
HIGH_SCORE_FILE = "duckhunt_scores.json"

# Colors (BGR for OpenCV)
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,   0,   0  )
C_RED     = (0,   0,   255)
C_GREEN   = (0,   255, 0  )
C_BLUE    = (255, 0,   0  )
C_YELLOW  = (0,   255, 255)
C_ORANGE  = (0,   165, 255)
C_CYAN    = (255, 255, 0  )
C_PINK    = (180, 105, 255)
C_GOLD    = (0,   215, 255)
C_SKY     = (235, 206, 135)   # sky blue
C_GRASS   = (34,  139, 34 )

# ─────────────────────────────────────────────
#  ENUMS
# ─────────────────────────────────────────────
class GameState(Enum):
    MENU        = "menu"
    INSTRUCTIONS= "instructions"
    PLAYING     = "playing"
    PAUSED      = "paused"
    LEVEL_UP    = "level_up"
    GAME_OVER   = "game_over"

class TargetType(Enum):
    DUCK    = "duck"     # normal
    FAST    = "fast"     # fast small duck
    BONUS   = "bonus"    # high-point gold star

# ─────────────────────────────────────────────
#  SOUND MANAGER
# ─────────────────────────────────────────────
class SoundManager:
    """Handles all audio: background music, gunshot, hit, game over."""

    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.sounds = {}
        self._generate_sounds()
        self.music_playing = False

    # --- Synthesise simple sounds procedurally (no external files needed) ---
    def _make_buffer(self, samples: np.ndarray):
        """Convert float32 [-1,1] samples → pygame.Sound."""
        samples = np.clip(samples, -1.0, 1.0)
        int16 = (samples * 32767).astype(np.int16)
        stereo = np.column_stack([int16, int16])
        return pygame.sndarray.make_sound(stereo)

    def _generate_sounds(self):
        sr = 44100

        # Gunshot – short broadband burst with pitch drop
        t  = np.linspace(0, 0.15, int(sr * 0.15))
        gs = np.random.uniform(-1, 1, len(t))  # white noise
        env = np.exp(-t * 40)
        gs  = gs * env * 0.8
        self.sounds["gunshot"] = self._make_buffer(gs)

        # Hit – ascending blip
        t   = np.linspace(0, 0.18, int(sr * 0.18))
        hit = np.sin(2 * np.pi * (600 + t * 1800) * t)
        hit *= np.exp(-t * 18) * 0.7
        self.sounds["hit"] = self._make_buffer(hit)

        # Miss – low thud
        t    = np.linspace(0, 0.20, int(sr * 0.20))
        miss = np.sin(2 * np.pi * 120 * t) * np.exp(-t * 20) * 0.5
        self.sounds["miss"] = self._make_buffer(miss)

        # Level up – cheerful arpeggio
        notes = [523, 659, 784, 1047]
        chunk = int(sr * 0.15)
        lu    = np.zeros(chunk * len(notes))
        for i, freq in enumerate(notes):
            tt = np.linspace(0, 0.15, chunk)
            lu[i*chunk:(i+1)*chunk] = np.sin(2*np.pi*freq*tt) * np.exp(-tt*8) * 0.6
        self.sounds["levelup"] = self._make_buffer(lu)

        # Game over – descending sad tone
        t  = np.linspace(0, 0.8, int(sr * 0.8))
        go = np.sin(2 * np.pi * (440 - t * 300) * t) * np.exp(-t * 3) * 0.6
        self.sounds["gameover"] = self._make_buffer(go)

        # Background music – simple 8-bit loop
        loop_len = int(sr * 2.0)
        bg = np.zeros(loop_len)
        melody = [262, 330, 392, 330, 262, 0, 294, 370, 440, 370]
        note_dur = loop_len // len(melody)
        for i, freq in enumerate(melody):
            if freq > 0:
                tt = np.linspace(0, note_dur/sr, note_dur)
                note = np.sin(2*np.pi*freq*tt) * 0.15
                note *= (1 - tt/(note_dur/sr)) * 0.8 + 0.2
                bg[i*note_dur:(i+1)*note_dur] += note
        self.sounds["background"] = self._make_buffer(bg)

    def play(self, name: str, loops: int = 0):
        if name in self.sounds:
            self.sounds[name].play(loops=loops)

    def start_music(self):
        if not self.music_playing:
            self.sounds["background"].play(loops=-1)
            self.music_playing = True

    def stop_music(self):
        if self.music_playing:
            self.sounds["background"].stop()
            self.music_playing = False

    def stop_all(self):
        pygame.mixer.stop()
        self.music_playing = False


# ─────────────────────────────────────────────
#  HAND TRACKER
# ─────────────────────────────────────────────
class HandTracker:
    """
    Wraps MediaPipe Hands.
    Detects: fingertip position, pinch gesture (shoot).
    """
    PINCH_THRESHOLD = 0.07   # normalised distance

    def __init__(self):
        self.mp_hands   = mp.solutions.hands
        self.hands      = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.mp_draw    = mp.solutions.drawing_utils
        self.fingertip  : Optional[Tuple[int,int]] = None
        self.is_pinching: bool   = False
        self._prev_pinch: bool   = False  # for edge detection

    def process(self, frame_rgb: np.ndarray, frame_w: int, frame_h: int):
        """Run inference. Update fingertip + pinch state."""
        results = self.hands.process(frame_rgb)
        self.fingertip = None
        self.is_pinching = False

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm   = hand.landmark

            # Index fingertip (id=8), thumb tip (id=4)
            ix = int(lm[8].x * frame_w)
            iy = int(lm[8].y * frame_h)
            tx = int(lm[4].x * frame_w)
            ty = int(lm[4].y * frame_h)

            # Mirror x (webcam is mirrored)
            self.fingertip = (frame_w - ix, iy)

            # Pinch distance (normalised)
            dist = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
            self.is_pinching = dist < self.PINCH_THRESHOLD

        return results

    @property
    def shot_fired(self) -> bool:
        """True only on the rising edge of a pinch."""
        fired = self.is_pinching and not self._prev_pinch
        self._prev_pinch = self.is_pinching
        return fired

    def reset(self):
        self._prev_pinch = False


# ─────────────────────────────────────────────
#  PARTICLE SYSTEM
# ─────────────────────────────────────────────
@dataclass
class Particle:
    x: float; y: float
    vx: float; vy: float
    life: float          # 0..1
    color: Tuple[int,int,int]
    size: int

class ParticleSystem:
    def __init__(self):
        self.particles: List[Particle] = []

    def emit_hit(self, x: int, y: int, color=(0,220,255)):
        for _ in range(20):
            angle = random.uniform(0, 2*math.pi)
            speed = random.uniform(2, 7)
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(angle)*speed,
                vy=math.sin(angle)*speed,
                life=1.0,
                color=color,
                size=random.randint(3, 7)
            ))

    def emit_shot(self, x: int, y: int):
        for _ in range(8):
            angle = random.uniform(0, 2*math.pi)
            speed = random.uniform(1, 4)
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(angle)*speed,
                vy=math.sin(angle)*speed,
                life=0.6,
                color=(50, 50, 200),
                size=random.randint(2, 4)
            ))

    def update(self, dt: float):
        decay = dt * 3.5
        self.particles = [
            p for p in self.particles if p.life > 0
        ]
        for p in self.particles:
            p.x   += p.vx
            p.y   += p.vy
            p.vy  += 0.15         # gravity
            p.life -= decay

    def draw(self, canvas: np.ndarray):
        for p in self.particles:
            alpha  = max(0.0, p.life)
            radius = max(1, int(p.size * alpha))
            color  = tuple(int(c * alpha) for c in p.color)
            cv2.circle(canvas, (int(p.x), int(p.y)), radius, color, -1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  GAME OBJECT (TARGET / DUCK)
# ─────────────────────────────────────────────
class GameObject:
    """
    A single target that moves across the screen.
    Supports duck, fast, bonus types.
    """
    # Type configs: (radius, speed_mult, score, color, wing_period)
    TYPE_CONFIG = {
        TargetType.DUCK : (30, 1.0,  10, (0, 165, 255), 0.25),
        TargetType.FAST : (20, 2.2,  20, (0,  50, 200), 0.15),
        TargetType.BONUS: (25, 0.6,  50, (0, 215, 255), 0.35),
    }

    def __init__(self, target_type: TargetType, screen_w: int, screen_h: int, level: int = 1):
        self.type   = target_type
        self.sw     = screen_w
        self.sh     = screen_h
        r, sm, score, color, wp = self.TYPE_CONFIG[target_type]
        self.radius = r
        self.score  = score
        self.color  = color
        self.wing_period = wp

        # Spawn on an edge (left or right)
        self.alive      = True
        self.hit        = False
        self.hit_timer  = 0.0
        self.spawn_time = time.time()

        speed_base  = (80 + level * 15) * sm
        self.vy     = random.uniform(-1.5, 1.5) * speed_base * 0.4
        if random.random() < 0.5:
            self.x  = -r
            self.vx = speed_base
        else:
            self.x  = screen_w + r
            self.vx = -speed_base

        self.y  = random.randint(80, int(screen_h * 0.72))
        self.t  = 0.0  # internal clock for animation

    def update(self, dt: float):
        if self.hit:
            self.hit_timer += dt
            self.y         += 180 * dt   # fall
            return

        self.t  += dt
        self.x  += self.vx * dt
        self.y  += self.vy * dt
        # Bounce vertically between sky and ground
        if self.y < 70 or self.y > self.sh * 0.78:
            self.vy = -self.vy

    def draw(self, canvas: np.ndarray):
        cx, cy = int(self.x), int(self.y)
        r = self.radius

        if self.hit:
            # Tumbling body
            alpha  = max(0, 1.0 - self.hit_timer / 0.6)
            c      = tuple(int(ch * alpha) for ch in self.color)
            cv2.circle(canvas, (cx, cy), r, c, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"+{self.score}", (cx-15, cy-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_GOLD, 2, cv2.LINE_AA)
            return

        # --- Draw body ---
        cv2.circle(canvas, (cx, cy), r, self.color, -1, cv2.LINE_AA)

        # --- Draw wings (animated) ---
        wing_y = int(math.sin(self.t * 2 * math.pi / self.wing_period) * r * 0.6)
        dir_x  = 1 if self.vx > 0 else -1
        # Left/right wings
        cv2.ellipse(canvas, (cx - dir_x * (r//2), cy + wing_y),
                    (r//2, r//3), 0, 0, 180, C_WHITE, -1, cv2.LINE_AA)
        # Eye
        eye_x = cx + dir_x * (r - 8)
        cv2.circle(canvas, (eye_x, cy - 4), 5, C_WHITE, -1, cv2.LINE_AA)
        cv2.circle(canvas, (eye_x + dir_x*2, cy - 4), 2, C_BLACK, -1)
        # Beak
        beak = [
            (eye_x + dir_x*6, cy - 2),
            (eye_x + dir_x*14, cy + 2),
            (eye_x + dir_x*6, cy + 6),
        ]
        cv2.fillPoly(canvas, [np.array(beak)], C_ORANGE)

        # Bonus star shimmer ring
        if self.type == TargetType.BONUS:
            ring_r = int(r + 6 + math.sin(self.t * 8) * 4)
            cv2.circle(canvas, (cx, cy), ring_r, C_GOLD, 2, cv2.LINE_AA)

    def check_hit(self, px: int, py: int) -> bool:
        if self.hit or not self.alive:
            return False
        dist = math.hypot(px - self.x, py - self.y)
        return dist <= self.radius + 10

    def is_offscreen(self) -> bool:
        return (self.x < -100 or self.x > self.sw + 100 or
                self.y > self.sh + 100)

    @property
    def should_remove(self) -> bool:
        return (self.hit and self.hit_timer > 0.7) or self.is_offscreen()


# ─────────────────────────────────────────────
#  BACKGROUND RENDERER
# ─────────────────────────────────────────────
class Background:
    """Draws a parallax sky + clouds + hills + grass scene."""

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.clouds: List[dict] = []
        for _ in range(6):
            self.clouds.append({
                "x": random.uniform(0, w),
                "y": random.uniform(30, h * 0.3),
                "speed": random.uniform(8, 22),
                "scale": random.uniform(0.6, 1.4),
            })
        self._base = self._build_base()

    def _build_base(self) -> np.ndarray:
        base = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Sky gradient
        for row in range(self.h):
            t   = row / self.h
            r   = int(135 + (34 - 135) * t)
            g   = int(206 + (139 - 206) * t)
            b   = int(235 + (34  - 235) * t)
            base[row, :] = (b, g, r)

        # Hills
        for hi, (xo, amp, col) in enumerate([
            (0,    0.18, (60, 120, 40)),
            (50,   0.14, (40, 100, 30)),
        ]):
            pts = []
            for x in range(self.w):
                y = int(self.h * 0.75 + math.sin((x + xo) / 120.0) * self.h * amp)
                pts.append((x, y))
            pts = [(0, self.h)] + pts + [(self.w, self.h)]
            cv2.fillPoly(base, [np.array(pts)], col)

        # Grass strip
        gh = int(self.h * 0.08)
        base[self.h - gh:, :] = (34, 120, 34)
        # Lighter grass top
        base[self.h - gh:self.h - gh + 5, :] = (50, 180, 50)

        return base

    def draw(self, canvas: np.ndarray, t: float):
        canvas[:] = self._base

        # Animate clouds
        for c in self.clouds:
            cx = int(c["x"]) % (self.w + 200) - 100
            cy = int(c["y"])
            sc = c["scale"]
            w_c = int(80 * sc)
            h_c = int(30 * sc)
            cv2.ellipse(canvas, (cx, cy), (w_c, h_c), 0, 0, 360,
                        C_WHITE, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx - w_c//3, cy + 5), (w_c//2, h_c//2),
                        0, 0, 360, C_WHITE, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx + w_c//3, cy + 5), (w_c//2, h_c//2),
                        0, 0, 360, C_WHITE, -1, cv2.LINE_AA)
            c["x"] += c["speed"] * (1/TARGET_FPS)
            if c["x"] > self.w + 100:
                c["x"] = -120


# ─────────────────────────────────────────────
#  HUD (Heads-Up Display)
# ─────────────────────────────────────────────
class HUD:
    def draw(self, canvas: np.ndarray, score: int, high_score: int,
             level: int, time_left: float, shots: int, hits: int,
             combo: int, is_pinching: bool):
        h, w = canvas.shape[:2]

        # Semi-transparent top bar
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), C_BLACK, -1)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

        # Score
        cv2.putText(canvas, f"SCORE: {score}", (15, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, C_YELLOW, 2, cv2.LINE_AA)
        # High score
        cv2.putText(canvas, f"BEST: {high_score}", (w//2 - 70, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, C_CYAN, 2, cv2.LINE_AA)
        # Level
        cv2.putText(canvas, f"LVL {level}", (w - 200, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, C_GREEN, 2, cv2.LINE_AA)
        # Timer
        color = C_RED if time_left < 10 else C_WHITE
        cv2.putText(canvas, f"{time_left:.0f}s", (w - 100, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)

        # Accuracy bottom left
        acc  = int(100 * hits / shots) if shots > 0 else 100
        cv2.putText(canvas, f"ACC: {acc}%", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 1, cv2.LINE_AA)

        # Combo indicator
        if combo >= 2:
            txt   = f"x{combo} COMBO!"
            scale = min(1.5, 0.8 + combo * 0.1)
            color = C_GOLD if combo >= 5 else C_ORANGE
            cv2.putText(canvas, txt, (w//2 - 70, h - 15),
                        cv2.FONT_HERSHEY_DUPLEX, scale, color, 2, cv2.LINE_AA)

        # Pinch indicator (gun ready / firing)
        if is_pinching:
            cv2.circle(canvas, (w - 40, h - 30), 14, C_RED, -1, cv2.LINE_AA)
            cv2.putText(canvas, "FIRE", (w - 110, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_RED, 2, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (w - 40, h - 30), 14, C_GREEN, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  CROSSHAIR
# ─────────────────────────────────────────────
def draw_crosshair(canvas: np.ndarray, x: int, y: int, is_pinching: bool):
    color  = C_RED   if is_pinching else C_WHITE
    size   = 22
    thick  = 2
    # Outer ring
    cv2.circle(canvas, (x, y), 18, color, thick, cv2.LINE_AA)
    # Cross lines
    cv2.line(canvas, (x - size, y), (x - 6, y), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x + 6,   y), (x + size, y), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x, y - size), (x, y - 6), color, thick, cv2.LINE_AA)
    cv2.line(canvas, (x, y + 6),    (x, y + size), color, thick, cv2.LINE_AA)
    # Center dot
    cv2.circle(canvas, (x, y), 3, color, -1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  FLASH EFFECT
# ─────────────────────────────────────────────
class FlashEffect:
    def __init__(self):
        self.alpha  = 0.0
        self.active = False

    def trigger(self):
        self.alpha  = 0.5
        self.active = True

    def update(self, dt: float):
        if self.active:
            self.alpha = max(0, self.alpha - dt * 4)
            if self.alpha <= 0:
                self.active = False

    def draw(self, canvas: np.ndarray):
        if self.active and self.alpha > 0:
            h, w = canvas.shape[:2]
            overlay = np.full((h, w, 3), 255, dtype=np.uint8)
            cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)


# ─────────────────────────────────────────────
#  HIGH SCORE MANAGER
# ─────────────────────────────────────────────
class HighScoreManager:
    def __init__(self, filepath: str = HIGH_SCORE_FILE):
        self.filepath  = filepath
        self.high_score = self._load()

    def _load(self) -> int:
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
                return int(data.get("high_score", 0))
        except Exception:
            return 0

    def update(self, score: int) -> bool:
        if score > self.high_score:
            self.high_score = score
            try:
                with open(self.filepath, "w") as f:
                    json.dump({"high_score": score}, f)
            except Exception:
                pass
            return True
        return False


# ─────────────────────────────────────────────
#  MENU SCREEN
# ─────────────────────────────────────────────
def draw_menu(canvas: np.ndarray, bg: Background, t: float, high_score: int,
              selected: int) -> None:
    """Draw the main menu over the background."""
    h, w = canvas.shape[:2]
    bg.draw(canvas, t)

    # Title card
    overlay = canvas.copy()
    cv2.rectangle(overlay, (w//2 - 320, 60), (w//2 + 320, 200), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    cv2.putText(canvas, "DUCK HUNT", (w//2 - 230, 130),
                cv2.FONT_HERSHEY_DUPLEX, 2.2, C_YELLOW, 4, cv2.LINE_AA)
    cv2.putText(canvas, "HAND GESTURE EDITION", (w//2 - 200, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, C_WHITE, 2, cv2.LINE_AA)

    # High score
    cv2.putText(canvas, f"BEST SCORE: {high_score}", (w//2 - 130, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, C_GOLD, 2, cv2.LINE_AA)

    options = ["  PLAY GAME  ", " INSTRUCTIONS", "    EXIT     "]
    for i, opt in enumerate(options):
        oy = 300 + i * 70
        color = C_YELLOW if i == selected else C_WHITE
        size  = 1.0     if i == selected else 0.85
        thick = 3       if i == selected else 2
        if i == selected:
            cv2.rectangle(canvas, (w//2 - 170, oy - 35),
                          (w//2 + 170, oy + 10), (0, 80, 0), -1)
        cv2.putText(canvas, opt, (w//2 - 145, oy),
                    cv2.FONT_HERSHEY_DUPLEX, size, color, thick, cv2.LINE_AA)

    # Instructions hint
    cv2.putText(canvas, "Pinch (thumb+index) over option to select",
                (w//2 - 245, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_CYAN, 1, cv2.LINE_AA)


def draw_instructions(canvas: np.ndarray, bg: Background, t: float):
    h, w = canvas.shape[:2]
    bg.draw(canvas, t)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (w//2 - 380, 50), (w//2 + 380, h - 50), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    lines = [
        ("HOW TO PLAY", 2.2, C_YELLOW, 115),
        ("", 0, C_WHITE, 0),
        ("AIMING", 1.0, C_CYAN,    170),
        ("Point your INDEX FINGER at the targets", 0.7, C_WHITE, 210),
        ("", 0, C_WHITE, 0),
        ("SHOOTING", 1.0, C_CYAN, 260),
        ("PINCH thumb + index finger together to SHOOT", 0.7, C_WHITE, 300),
        ("", 0, C_WHITE, 0),
        ("TARGETS", 1.0, C_CYAN, 350),
        ("Orange duck  = 10 pts   |  Fast blue = 20 pts", 0.7, C_WHITE, 390),
        ("Gold bonus   = 50 pts   |  Hit 3+ in a row for COMBO!", 0.7, C_WHITE, 425),
        ("", 0, C_WHITE, 0),
        ("LEVELS", 1.0, C_CYAN,  475),
        ("Every 5 hits = new level. Targets get faster!", 0.7, C_WHITE, 515),
        ("", 0, C_WHITE, 0),
        ("KEYS", 1.0, C_CYAN,  565),
        ("P = Pause/Resume    Q = Quit", 0.7, C_WHITE, 605),
        ("", 0, C_WHITE, 0),
        ("Pinch to go BACK", 0.8, C_YELLOW, h - 70),
    ]
    for text, scale, color, y in lines:
        if text and scale > 0:
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)[0][0]
            cv2.putText(canvas, text, (w//2 - tw//2, y),
                        cv2.FONT_HERSHEY_DUPLEX, scale, color, 2, cv2.LINE_AA)


def draw_game_over(canvas: np.ndarray, bg: Background, t: float,
                   score: int, high_score: int, is_new_record: bool):
    h, w = canvas.shape[:2]
    bg.draw(canvas, t)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (w//2 - 310, 100), (w//2 + 310, h - 100), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    cv2.putText(canvas, "GAME OVER", (w//2 - 220, 185),
                cv2.FONT_HERSHEY_DUPLEX, 2.0, C_RED, 4, cv2.LINE_AA)

    cv2.putText(canvas, f"YOUR SCORE: {score}", (w//2 - 165, 265),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, C_WHITE, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"BEST SCORE: {high_score}", (w//2 - 165, 315),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, C_GOLD, 2, cv2.LINE_AA)

    if is_new_record:
        cv2.putText(canvas, "** NEW RECORD! **", (w//2 - 185, 375),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, C_YELLOW, 3, cv2.LINE_AA)

    cv2.putText(canvas, "Pinch to PLAY AGAIN", (w//2 - 185, h - 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_GREEN, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Hold pinch 1s for MENU", (w//2 - 185, h - 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_CYAN, 2, cv2.LINE_AA)


def draw_level_up(canvas: np.ndarray, level: int):
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (w//2 - 250, h//2 - 70),
                  (w//2 + 250, h//2 + 70), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

    cv2.putText(canvas, f"LEVEL {level}!", (w//2 - 120, h//2 + 5),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, C_YELLOW, 4, cv2.LINE_AA)
    cv2.putText(canvas, "Get ready...", (w//2 - 100, h//2 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_WHITE, 2, cv2.LINE_AA)


def draw_paused(canvas: np.ndarray):
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (w//2 - 180, h//2 - 60),
                  (w//2 + 180, h//2 + 60), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    cv2.putText(canvas, "PAUSED", (w//2 - 105, h//2 + 10),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, C_YELLOW, 3, cv2.LINE_AA)
    cv2.putText(canvas, "Press P to resume", (w//2 - 140, h//2 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_WHITE, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  GAME ENGINE
# ─────────────────────────────────────────────
class GameEngine:
    """
    Central game controller.
    Manages state machine, spawning, scoring, input from HandTracker.
    """
    ROUND_DURATION  = 60.0     # seconds per round
    LEVEL_HITS      = 5        # hits to advance a level
    MAX_TARGETS     = 6
    SPAWN_INTERVAL  = 2.0      # base seconds between spawns

    def __init__(self):
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, frame = self.cap.read()
        if not ret:
            print("ERROR: Could not open webcam.")
            sys.exit(1)
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Subsystems
        self.tracker    = HandTracker()
        self.sound      = SoundManager()
        self.bg         = Background(self.W, self.H)
        self.particles  = ParticleSystem()
        self.flash      = FlashEffect()
        self.hud        = HUD()
        self.hs_manager = HighScoreManager()

        # Game state
        self.state      = GameState.MENU
        self.score      = 0
        self.level      = 1
        self.shots      = 0
        self.hits       = 0
        self.combo      = 0
        self.max_combo  = 0
        self.time_left  = self.ROUND_DURATION
        self.targets   : List[GameObject] = []
        self.spawn_t    = 0.0
        self.level_up_t = 0.0
        self.hit_count_level = 0

        # Menu state
        self.menu_sel   = 0
        self.pinch_held = 0.0    # seconds pinch held (for hold-to-select)
        self.go_pinch_t = 0.0    # game-over hold timer
        self.is_new_record = False
        self.t_global   = 0.0
        self.prev_time  = time.time()

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, self.W, self.H)

    # ── Main loop ─────────────────────────────
    def run(self):
        self.sound.start_music()
        while True:
            now = time.time()
            dt  = min(now - self.prev_time, 0.05)
            self.prev_time = now
            self.t_global += dt

            ret, raw_frame = self.cap.read()
            if not ret:
                break

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            self.tracker.process(rgb, self.W, self.H)

            canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            self._dispatch(canvas, dt)

            cv2.imshow(WINDOW_TITLE, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            self._handle_keys(key)

        self._cleanup()

    # ── Key handling ──────────────────────────
    def _handle_keys(self, key: int):
        if key in (ord('p'), ord('P')):
            if self.state == GameState.PLAYING:
                self.state = GameState.PAUSED
            elif self.state == GameState.PAUSED:
                self.state = GameState.PLAYING
                self.prev_time = time.time()

    # ── State dispatcher ──────────────────────
    def _dispatch(self, canvas: np.ndarray, dt: float):
        s = self.state
        if s == GameState.MENU:
            self._state_menu(canvas, dt)
        elif s == GameState.INSTRUCTIONS:
            self._state_instructions(canvas, dt)
        elif s == GameState.PLAYING:
            self._state_playing(canvas, dt)
        elif s == GameState.PAUSED:
            self._state_paused(canvas, dt)
        elif s == GameState.LEVEL_UP:
            self._state_level_up(canvas, dt)
        elif s == GameState.GAME_OVER:
            self._state_game_over(canvas, dt)

    # ── MENU ──────────────────────────────────
    def _state_menu(self, canvas: np.ndarray, dt: float):
        draw_menu(canvas, self.bg, self.t_global,
                  self.hs_manager.high_score, self.menu_sel)

        tip = self.tracker.fingertip
        if tip:
            draw_crosshair(canvas, tip[0], tip[1], self.tracker.is_pinching)
            # Map fingertip to option zones (y ranges)
            y_zones = [(265, 340), (335, 405), (405, 475)]
            for i, (y0, y1) in enumerate(y_zones):
                if y0 <= tip[1] <= y1:
                    self.menu_sel = i
                    break

            # Pinch-hold to select
            if self.tracker.is_pinching:
                self.pinch_held += dt
                # Show hold progress
                prog = min(1.0, self.pinch_held / 0.8)
                cv2.rectangle(canvas,
                              (self.W//2 - 150, self.H - 80),
                              (int(self.W//2 - 150 + 300 * prog), self.H - 60),
                              C_GREEN, -1)
                cv2.rectangle(canvas,
                              (self.W//2 - 150, self.H - 80),
                              (self.W//2 + 150, self.H - 60),
                              C_WHITE, 2)
                if self.pinch_held >= 0.8:
                    self.pinch_held = 0
                    self._select_menu(self.menu_sel)
            else:
                self.pinch_held = 0

    def _select_menu(self, idx: int):
        if idx == 0:
            self._start_game()
        elif idx == 1:
            self.state = GameState.INSTRUCTIONS
        elif idx == 2:
            self._cleanup(); sys.exit(0)

    # ── INSTRUCTIONS ─────────────────────────
    def _state_instructions(self, canvas: np.ndarray, dt: float):
        draw_instructions(canvas, self.bg, self.t_global)
        tip = self.tracker.fingertip
        if tip:
            draw_crosshair(canvas, tip[0], tip[1], self.tracker.is_pinching)
        if self.tracker.shot_fired:
            self.state = GameState.MENU

    # ── PLAYING ───────────────────────────────
    def _state_playing(self, canvas: np.ndarray, dt: float):
        # Update time
        self.time_left -= dt
        if self.time_left <= 0:
            self._end_game()
            return

        # Spawn targets
        self.spawn_t += dt
        spawn_interval = max(0.6, self.SPAWN_INTERVAL - self.level * 0.12)
        if self.spawn_t >= spawn_interval and len(self.targets) < self.MAX_TARGETS:
            self.spawn_t = 0
            self._spawn_target()

        # Update targets
        for tgt in self.targets[:]:
            tgt.update(dt)
            if tgt.should_remove and not tgt.hit:
                # Missed duck
                self.combo = 0
            if tgt.should_remove:
                self.targets.remove(tgt)

        # Particles & flash
        self.particles.update(dt)
        self.flash.update(dt)

        # Draw
        self.bg.draw(canvas, self.t_global)
        for tgt in self.targets:
            tgt.draw(canvas)
        self.particles.draw(canvas)
        self.flash.draw(canvas)

        # Gesture → shoot
        tip = self.tracker.fingertip
        if tip:
            draw_crosshair(canvas, tip[0], tip[1], self.tracker.is_pinching)
            if self.tracker.shot_fired:
                self._process_shot(tip[0], tip[1])

        self.hud.draw(canvas, self.score, self.hs_manager.high_score,
                      self.level, self.time_left, self.shots, self.hits,
                      self.combo, self.tracker.is_pinching)

    def _spawn_target(self):
        r = random.random()
        if r < 0.60:
            ttype = TargetType.DUCK
        elif r < 0.85:
            ttype = TargetType.FAST
        else:
            ttype = TargetType.BONUS
        self.targets.append(
            GameObject(ttype, self.W, self.H, self.level)
        )

    def _process_shot(self, px: int, py: int):
        self.shots += 1
        self.flash.trigger()
        self.particles.emit_shot(px, py)
        self.sound.play("gunshot")

        hit_any = False
        for tgt in self.targets:
            if tgt.check_hit(px, py):
                tgt.hit      = True
                tgt.hit_timer = 0
                self.hits    += 1
                self.combo   += 1
                self.max_combo = max(self.max_combo, self.combo)
                combo_bonus  = max(1, self.combo // 2)
                pts          = tgt.score * combo_bonus
                self.score  += pts
                self.hit_count_level += 1
                self.sound.play("hit")
                self.particles.emit_hit(px, py, tgt.color)
                hit_any = True

                # Level up?
                if self.hit_count_level >= self.LEVEL_HITS:
                    self.hit_count_level = 0
                    self.level          += 1
                    self.state           = GameState.LEVEL_UP
                    self.level_up_t      = 0.0
                    self.sound.play("levelup")
                break

        if not hit_any:
            self.combo = 0
            self.sound.play("miss")

    # ── LEVEL UP ──────────────────────────────
    def _state_level_up(self, canvas: np.ndarray, dt: float):
        self.bg.draw(canvas, self.t_global)
        for tgt in self.targets:
            tgt.draw(canvas)
        draw_level_up(canvas, self.level)
        self.level_up_t += dt
        if self.level_up_t >= 2.2:
            self.state = GameState.PLAYING

    # ── PAUSED ────────────────────────────────
    def _state_paused(self, canvas: np.ndarray, dt: float):
        self.bg.draw(canvas, self.t_global)
        for tgt in self.targets:
            tgt.draw(canvas)
        draw_paused(canvas)

    # ── GAME OVER ─────────────────────────────
    def _state_game_over(self, canvas: np.ndarray, dt: float):
        draw_game_over(canvas, self.bg, self.t_global,
                       self.score, self.hs_manager.high_score, self.is_new_record)

        tip = self.tracker.fingertip
        if tip:
            draw_crosshair(canvas, tip[0], tip[1], self.tracker.is_pinching)

        # Quick pinch → restart, hold pinch → menu
        if self.tracker.is_pinching:
            self.go_pinch_t += dt
            prog = min(1.0, self.go_pinch_t / 1.2)
            cv2.rectangle(canvas,
                          (self.W//2 - 150, self.H - 80),
                          (int(self.W//2 - 150 + 300 * prog), self.H - 60),
                          C_CYAN, -1)
            cv2.rectangle(canvas, (self.W//2 - 150, self.H - 80),
                          (self.W//2 + 150, self.H - 60), C_WHITE, 2)
            cv2.putText(canvas, "Hold for MENU", (self.W//2 - 90, self.H - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 1, cv2.LINE_AA)
            if self.go_pinch_t >= 1.2:
                self.go_pinch_t = 0
                self.state = GameState.MENU
        else:
            if self.go_pinch_t > 0.05 and self.go_pinch_t < 1.0:
                self._start_game()
            self.go_pinch_t = 0

    # ── Game management ───────────────────────
    def _start_game(self):
        self.score          = 0
        self.level          = 1
        self.shots          = 0
        self.hits           = 0
        self.combo          = 0
        self.max_combo      = 0
        self.time_left      = self.ROUND_DURATION
        self.targets.clear()
        self.particles      = ParticleSystem()
        self.spawn_t        = 0
        self.hit_count_level = 0
        self.is_new_record  = False
        self.tracker.reset()
        self.state          = GameState.PLAYING
        self.prev_time      = time.time()
        self.sound.start_music()

    def _end_game(self):
        self.state         = GameState.GAME_OVER
        self.is_new_record = self.hs_manager.update(self.score)
        self.sound.stop_music()
        self.sound.play("gameover")
        self.go_pinch_t = 0

    def _cleanup(self):
        self.cap.release()
        self.sound.stop_all()
        cv2.destroyAllWindows()
        pygame.quit()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DUCK HUNT CV  –  Hand Gesture Edition")
    print("=" * 60)
    print("  Controls:")
    print("    • Point index finger to aim")
    print("    • Pinch (thumb + index) to SHOOT")
    print("    • P = Pause   Q = Quit")
    print("=" * 60)
    print("  Starting webcam...")

    pygame.init()
    engine = GameEngine()
    engine.run()


if __name__ == "__main__":
    main()

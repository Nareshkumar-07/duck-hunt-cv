Here's a great README for your project:

---

# 🦆 Duck Hunt CV — Hand Gesture Edition

A real-time **Duck Hunt** game controlled entirely by **hand gestures** using your webcam. No mouse, no keyboard — just point and shoot with your fingers!

Built with **OpenCV**, **MediaPipe**, and **Pygame**.

---

## 🎮 Demo
> Point your index finger to aim. Pinch thumb + index finger to shoot!

---

## 🚀 Features
- 👆 Real-time hand tracking via webcam
- 🎯 Index finger as crosshair/aim
- 🤏 Pinch gesture to shoot
- 🦆 3 target types — Normal, Fast, and Bonus
- 🔥 Combo scoring system
- 📈 Level progression (gets harder over time)
- ⏱️ 60-second timed rounds
- 🏆 Local high score tracking
- 💥 Particle effects and flash animations
- 🔊 Procedurally generated sound effects
- ⏸️ Pause/Resume support

---

## 🛠️ Tech Stack
| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| OpenCV | Camera input & rendering |
| MediaPipe | Hand tracking |
| Pygame | Audio playback |
| NumPy | Array processing |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Nareshkumar-07/duck-hunt-cv.git
cd duck-hunt-cv

# Install dependencies
pip install opencv-python mediapipe pygame numpy
```

---

## ▶️ Run the Game

```bash
py -3.12 duck_hunt_cv.py
```

---

## 🕹️ Controls
| Action | Gesture / Key |
|---|---|
| Aim | Point index finger |
| Shoot | Pinch thumb + index finger |
| Pause / Resume | Press `P` |
| Quit | Press `Q` |

---

## 🎯 Target Types
| Target | Speed | Points |
|---|---|---|
| 🟠 Orange Duck | Normal | 10 pts |
| 🔵 Fast Blue | 2x Fast | 20 pts |
| ✨ Gold Bonus | Slow | 50 pts |

---

## 📁 Project Structure
```
duck-hunt-cv/
├── duck_hunt_cv.py      # Main game file
├── requirements.txt     # Dependencies
└── README.md
```

---

## 👨‍💻 Author
**Naresh Kumar** — [@Nareshkumar-07](https://github.com/Nareshkumar-07)

---

## ⭐ Support
If you like this project, give it a **star** on GitHub! ⭐
---

import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

# ================== INITIALIZATION ==================
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

canvas = None
undo_stack = deque(maxlen=20)
redo_stack = deque(maxlen=20)

prev_x, prev_y = 0, 0
show_help = False

# ================== LOAD LOGO ==================
logo = cv2.imread("logo.png", cv2.IMREAD_UNCHANGED)
if logo is not None:
    logo = cv2.resize(logo, (160, 160))

def overlay_logo(frame, logo, x, y):
    h, w = logo.shape[:2]
    if logo.shape[2] == 4:  # PNG with alpha
        for c in range(3):
            # Check bounds to prevent error
            if y+h < frame.shape[0] and x+w < frame.shape[1]:
                frame[y:y+h, x:x+w, c] = (
                    logo[:, :, c] * (logo[:, :, 3] / 255.0) +
                    frame[y:y+h, x:x+w, c] * (1.0 - logo[:, :, 3] / 255.0)
                )
    else:
        if y+h < frame.shape[0] and x+w < frame.shape[1]:
            frame[y:y+h, x:x+w] = logo

# ================== COLORS ==================
colors = [
    (255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255),
    (211, 0, 148), (203, 192, 255), (235, 206, 135), (0, 0, 0)
]
current_color = colors[0]

# ================== BRUSH SETTINGS ==================
brush_size = 8
MIN_BRUSH, MAX_BRUSH = 3, 40

brush_types = {
    "PENCIL": 0.5,
    "BRUSH": 1.0,
    "MARKER": 1.8,
    "ERASER": 3.0
}
current_brush = "BRUSH"

# ================== UI FUNCTIONS ==================
def draw_color_buttons(img):
    # NOTE: cx, cy, and radius must match the detection logic
    for i, col in enumerate(colors):
        cx = 60 + i * 70
        cy = 80
        cv2.circle(img, (cx, cy), 22, col, cv2.FILLED)
        if col == current_color:
            cv2.circle(img, (cx, cy), 25, (255, 255, 255), 2)

def draw_help_button(img, w):
    cx, cy = w - 80, 140  # spaced from eraser
    cv2.circle(img, (cx, cy), 25, (40, 40, 40), cv2.FILLED)
    cv2.putText(img, "?", (cx - 8, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    return cx, cy

def draw_help_panel(img):
    overlay = img.copy()
    cv2.rectangle(overlay, (100, 100), (760, 500), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)

    lines = [
        "HELP - GESTURES & KEYS",
        "",
        "Draw              : Index finger up",
        "Select Color      : Index + Middle fingers (point at color)", # Updated hint
        "Change Brush Size : Thumb - Index pinch",
        "Change Brush Type : Keys 1 - 4",
        "Undo              : Z",
        "Redo              : Y",
        "Clear Canvas      : C",
        "Exit              : Q / ESC"
    ]

    y = 150
    for line in lines:
        cv2.putText(img, line, (120, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
        y += 35

def draw_title(img, w):
    overlay = img.copy()
    cv2.rectangle(overlay, (w//2 - 150, 10), (w//2 + 60 , 55), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.9, img, 0.4, 0)
    cv2.putText(img, "AirDoodle",
                (w//2 - 130, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 3)

def check_circle(x, y, cx, cy, r):
    return math.hypot(x - cx, y - cy) < r

# ================== HAND HELPERS ==================
def fingers_up(hand):
    index_up = hand.landmark[8].y < hand.landmark[6].y
    middle_up = hand.landmark[12].y < hand.landmark[10].y
    return index_up, middle_up

def pinch_distance(hand, w, h):
    x1 = int(hand.landmark[4].x * w)
    y1 = int(hand.landmark[4].y * h)
    x2 = int(hand.landmark[8].x * w)
    y2 = int(hand.landmark[8].y * h)
    return math.hypot(x2 - x1, y2 - y1)

# ================== FULL SCREEN ==================
window = "AirDoodle"
cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ================== MAIN LOOP ==================
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    draw_color_buttons(frame)
    draw_title(frame, w)

    # Draw logo
    if logo is not None:
        overlay_logo(frame, logo, w//2 - logo.shape[1]//2, 70)

    help_cx, help_cy = draw_help_button(frame, w)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS) # 

            index_up, middle_up = fingers_up(hand)
            # x, y are the coordinates of the Index Finger Tip (landmark 8)
            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)

            # SELECTION MODE: Index Up AND Middle Up
            if index_up and middle_up:
                
                # --- COLOR SELECTION LOGIC ADDED HERE ---
                start_cx = 60
                button_spacing = 70
                button_radius = 25 # Must match the radius used in draw_color_buttons
                button_cy = 80    # Must match the Y position used in draw_color_buttons
                
                for i, col in enumerate(colors):
                    cx = start_cx + i * button_spacing
                    
                    if check_circle(x, y, cx, button_cy, button_radius):
                        current_color = col
                        # Switch brush type if black is selected
                        if current_color == (0, 0, 0):
                            current_brush = "ERASER"
                        elif current_brush == "ERASER":
                             current_brush = "BRUSH" # Switch back to default brush
                        cv2.waitKey(200) # Debounce the selection
                        break
                
                # --- HELP BUTTON LOGIC ---
                if check_circle(x, y, help_cx, help_cy, 25):
                    show_help = not show_help
                    cv2.waitKey(300)

            # BRUSH SIZE CONTROL: Thumb-Index Pinch Distance
            dist = pinch_distance(hand, w, h)
            if dist < 40:
                brush_size = max(MIN_BRUSH, brush_size - 1)
            elif dist > 90:
                brush_size = min(MAX_BRUSH, brush_size + 1)

            # DRAWING MODE: Index Up ONLY
            if index_up and not middle_up and not show_help:
                if prev_x == 0:
                    undo_stack.append(canvas.copy())
                    redo_stack.clear()
                    prev_x, prev_y = x, y

                thickness = int(brush_size * brush_types[current_brush])
                color = (0, 0, 0) if current_brush == "ERASER" else current_color

                cv2.line(canvas, (prev_x, prev_y), (x, y),
                         color, thickness, cv2.LINE_AA)
                prev_x, prev_y = x, y
            else:
                prev_x = 0

    # Merge canvas
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    if show_help:
        draw_help_panel(frame)

    cv2.imshow(window, frame)

    key = cv2.waitKey(1) & 0xFF
    # Existing keyboard controls...
    if key == ord('1'): 
        current_brush = "PENCIL"
        if current_color == (0, 0, 0): current_color = colors[0] # Switch color if black was eraser
    if key == ord('2'): 
        current_brush = "BRUSH"
        if current_color == (0, 0, 0): current_color = colors[0]
    if key == ord('3'): 
        current_brush = "MARKER"
        if current_color == (0, 0, 0): current_color = colors[0]
    if key == ord('4'): current_brush = "ERASER"

    if key == ord('z') and undo_stack:
        redo_stack.append(canvas.copy())
        canvas = undo_stack.pop()

    if key == ord('y') and redo_stack:
        undo_stack.append(canvas.copy())
        canvas = redo_stack.pop()

    if key == ord('c'):
        undo_stack.append(canvas.copy())
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if key == ord('q') or key == 27:
        break

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()
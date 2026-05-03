"""
✋ VIRTUAL HAND CONTROL SYSTEM  v3.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALL:
    pip install mediapipe opencv-python pyautogui numpy pycaw screen-brightness-control comtypes

RUN:
    python virtual_hand_control.py

GESTURES:
    ☝️  Index only            → Move cursor (ultra-smooth)
    ✌️  Index+Middle close    → Left Click
    🤏  Thumb+Index pinch     → Right Click
    🖐️  All 5 fingers         → Scroll  (hand up/down)
    🤘  Index+Pinky           → Volume  (hand up/down)
    👌  Mid+Ring+Pinky        → Brightness (hand up/down)
    🤙  Thumb+Pinky           → Toggle Keyboard ON/OFF
    ✊  Fist                  → Freeze cursor

KEYBOARD:
    Hover index over key  → key highlights
    Pinch thumb+index     → types key

Press Q to quit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math

# ── Optional: Volume (Windows) ─────────────────────────────────
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    _dev   = AudioUtilities.GetSpeakers()
    _iface = _dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_obj       = cast(_iface, POINTER(IAudioEndpointVolume))
    VOL_MIN, VOL_MAX = volume_obj.GetVolumeRange()[:2]
    VOLUME_OK        = True
except Exception:
    VOLUME_OK = False

# ── Optional: Brightness ────────────────────────────────────────
try:
    import screen_brightness_control as sbc
    BRIGHT_OK = True
except Exception:
    BRIGHT_OK = False

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
CAMERA_ID      = 0
FRAME_W        = 1280
FRAME_H        = 720

# Mouse
CLICK_DIST     = 36
PINCH_DIST     = 36
CLICK_COOLDOWN = 0.35
FRAME_RED      = 90       # detection zone inset from edges

# One-Euro cursor smoother  (tune these two)
SMOOTH_MIN_CUTOFF = 0.8   # lower  = smoother at rest
SMOOTH_BETA       = 0.06  # higher = snappier on fast moves
SMOOTH_D_CUTOFF   = 1.0

# Scroll / Vol / Bright
SCROLL_SPEED   = 3
VOL_SENS       = 0.25
BRIGHT_SENS    = 0.25

# Keyboard
KEY_GAP        = 6
KB_MARGIN      = 12
KB_PADDING     = 14
KB_HEIGHT_PCT  = 0.42     # keyboard = 42% frame height
TEXT_BAR_H     = 46
PINCH_KB_DIST  = 40
KEY_COOLDOWN   = 0.28

# ── Key colours (BGR) ──────────────────────────────────────────
KEY_BG_NORMAL  = (220, 220, 220)   # light grey  – resting key face
KEY_BG_HOVER   = (255, 255, 255)   # pure white  – hovered key
KEY_BORDER     = (90,  90,  90)    # dark border always visible
KEY_TXT_NORMAL = (20,  20,  20)    # near-black text on grey key
KEY_TXT_HOVER  = (0,   0,   0)     # pure black  on white hovered key
PANEL_BG       = (40,  40,  40)    # dark panel background
TEXTBAR_BG     = (60,  60,  60)    # slightly lighter text bar

SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.FAILSAFE  = False
pyautogui.PAUSE     = 0

# ══════════════════════════════════════════════════════════════
#  MEDIAPIPE
# ══════════════════════════════════════════════════════════════
mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands_model = mp_hands.Hands(
    static_image_mode        = False,
    max_num_hands            = 1,
    min_detection_confidence = 0.80,
    min_tracking_confidence  = 0.80,
)

KB_ROWS = [
    ['Q','W','E','R','T','Y','U','I','O','P'],
    ['A','S','D','F','G','H','J','K','L',';'],
    ['Z','X','C','V','B','N','M',',','.','<'],
    ['SPACE','ENTER'],
]

# ══════════════════════════════════════════════════════════════
#  ONE-EURO FILTER
# ══════════════════════════════════════════════════════════════
class OneEuroFilter:
    def __init__(self, min_cutoff=SMOOTH_MIN_CUTOFF,
                 beta=SMOOTH_BETA, d_cutoff=SMOOTH_D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_prev     = None
        self.dx_prev    = 0.0
        self.t_prev     = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt     = max(t - self.t_prev, 1e-6)
        dx     = (x - self.x_prev) / dt
        a_d    = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a      = self._alpha(cutoff, dt)
        x_hat  = a * x + (1 - a) * self.x_prev
        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
        return x_hat


class SmoothCursor:
    def __init__(self):
        self.fx = OneEuroFilter()
        self.fy = OneEuroFilter()

    def smooth(self, x, y):
        t = time.time()
        return int(self.fx(x, t)), int(self.fy(y, t))


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def get_landmarks(frame, results):
    if not results.multi_hand_landmarks:
        return None
    h, w = frame.shape[:2]
    lm   = results.multi_hand_landmarks[0].landmark
    return [(int(p.x * w), int(p.y * h)) for p in lm]


def fingers_up(pts):
    if pts is None:
        return [False] * 5
    tips = [4,  8, 12, 16, 20]
    pips = [3,  6, 10, 14, 18]
    up   = [pts[tips[0]][0] < pts[pips[0]][0]]
    for i in range(1, 5):
        up.append(pts[tips[i]][1] < pts[pips[i]][1])
    return up


def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def map_to_screen(x, y, fw, fh):
    sx = np.interp(x, [FRAME_RED, fw-FRAME_RED], [0, SCREEN_W])
    sy = np.interp(y, [FRAME_RED, fh-FRAME_RED], [0, SCREEN_H])
    return float(sx), float(sy)


def detect_gesture(up, pts):
    if pts is None:
        return "NO_HAND"
    t, i, m, r, p = up
    if t  and not i and not m and not r and p:  return "KB_TOGGLE"
    if not t and i  and not m and not r and p:  return "VOLUME"
    if not t and not i and m  and r  and p:     return "BRIGHTNESS"
    if all(up):                                 return "SCROLL"
    if not any(up):                             return "PAUSE"
    if i and m and not r and not p:
        return "LEFT_CLICK" if dist(pts[8], pts[12]) < CLICK_DIST else "MOVE"
    if t and i and not m and not r and not p:
        return "RIGHT_CLICK" if dist(pts[4], pts[8]) < PINCH_DIST else "MOVE"
    if i and not m:                             return "MOVE"
    return "PAUSE"


# ══════════════════════════════════════════════════════════════
#  KEYBOARD
# ══════════════════════════════════════════════════════════════
def build_key_rects(fw, fh):
    num_rows    = len(KB_ROWS)
    max_keys    = max(len(r) for r in KB_ROWS)
    panel_x1    = KB_MARGIN
    panel_x2    = fw - KB_MARGIN
    panel_w     = panel_x2 - panel_x1
    kb_total_h  = int(fh * KB_HEIGHT_PCT)
    keys_area_h = kb_total_h - TEXT_BAR_H - KB_PADDING * 2 - 10
    key_w       = (panel_w - KB_PADDING*2 - KEY_GAP*(max_keys-1)) // max_keys
    key_h       = (keys_area_h - KEY_GAP*(num_rows-1)) // num_rows
    panel_y1    = fh - kb_total_h - KB_MARGIN
    panel_y2    = fh - KB_MARGIN
    keys_start  = panel_y1 + KB_PADDING + TEXT_BAR_H + 8

    keys = []
    for ri, row in enumerate(KB_ROWS):
        y = keys_start + ri * (key_h + KEY_GAP)
        if row == ['SPACE', 'ENTER']:
            sp_w  = key_w * 6 + KEY_GAP * 5
            en_w  = key_w * 3 + KEY_GAP * 2
            tot_w = sp_w + en_w + KEY_GAP
            rx    = panel_x1 + KB_PADDING + (panel_w - KB_PADDING*2 - tot_w) // 2
            keys.append(('SPACE', rx, y, rx+sp_w, y+key_h))
            ex = rx + sp_w + KEY_GAP
            keys.append(('ENTER', ex, y, ex+en_w, y+key_h))
        else:
            tot_w = len(row)*key_w + (len(row)-1)*KEY_GAP
            rx    = panel_x1 + KB_PADDING + (panel_w - KB_PADDING*2 - tot_w) // 2
            for key in row:
                keys.append((key, rx, y, rx+key_w, y+key_h))
                rx += key_w + KEY_GAP

    return keys, (panel_x1, panel_y1, panel_x2, panel_y2), key_w, key_h


def draw_keyboard(frame, key_rects, panel_rect, hovered_key, typed_text):
    px1, py1, px2, py2 = panel_rect

    # ── Panel: semi-transparent dark background ───────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)   # 45% dark = see-through
    cv2.rectangle(frame, (px1, py1), (px2, py2), (130, 130, 130), 1)

    # ── Typed text bar ───────────────────────────────────────────
    tb_x1 = px1 + KB_PADDING
    tb_y1 = py1 + 8
    tb_x2 = px2 - KB_PADDING
    tb_y2 = py1 + 8 + TEXT_BAR_H
    tb_ov = frame.copy()
    cv2.rectangle(tb_ov, (tb_x1, tb_y1), (tb_x2, tb_y2), TEXTBAR_BG, -1)
    cv2.addWeighted(tb_ov, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (tb_x1, tb_y1), (tb_x2, tb_y2), (160, 160, 160), 1)

    blink = "|" if int(time.time()*2) % 2 == 0 else " "
    disp  = (typed_text[-58:] if len(typed_text) > 58 else typed_text) + blink
    cv2.putText(frame, disp, (tb_x1+10, tb_y1+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (240, 240, 240), 1, cv2.LINE_AA)

    # ── Keys: blend fill for transparency, then draw border & text directly ──
    key_ov = frame.copy()
    for (key, x1, y1, x2, y2) in key_rects:
        hov    = (key == hovered_key)
        bg_col = KEY_BG_HOVER if hov else KEY_BG_NORMAL
        cv2.rectangle(key_ov, (x1, y1), (x2, y2), bg_col, -1)

    # Blend keys at 35% — semi-transparent, camera shows through
    cv2.addWeighted(key_ov, 0.35, frame, 0.65, 0, frame)

    # Borders & text drawn AFTER blend so they are always crisp & fully opaque
    for (key, x1, y1, x2, y2) in key_rects:
        hov     = (key == hovered_key)
        bdr_col = (0, 210, 150) if hov else (200, 200, 200)
        txt_col = (0, 0, 0)              # BLACK letters on every key
        bdr_w   = 2 if hov else 1
        thick   = 2 if hov else 2        # always bold so text is readable

        cv2.rectangle(frame, (x1, y1), (x2, y2), bdr_col, bdr_w)

        label = key
        fs    = 0.38 if key in ('SPACE', 'ENTER') else 0.52
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        cx = x1 + (x2-x1)//2 - tw//2
        cy = y1 + (y2-y1)//2 + th//2

        # White outline/shadow behind text for contrast against any background
        cv2.putText(frame, label, (cx+1, cy+1),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick+1, cv2.LINE_AA)
        # Black text on top
        cv2.putText(frame, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, txt_col, thick, cv2.LINE_AA)


def get_hovered_key(pt, key_rects):
    fx, fy = pt
    for (key, x1, y1, x2, y2) in key_rects:
        if x1 < fx < x2 and y1 < fy < y2:
            return key
    return None


def press_key(key, text):
    if key == '<':
        pyautogui.press('backspace')
        return text[:-1]
    elif key == 'SPACE':
        pyautogui.press('space')
        return text + ' '
    elif key == 'ENTER':
        pyautogui.press('enter')
        return text + '\n'
    else:
        pyautogui.typewrite(key.lower(), interval=0)
        return text + key


# ══════════════════════════════════════════════════════════════
#  HUD
# ══════════════════════════════════════════════════════════════
GESTURE_INFO = {
    "MOVE"       : ("MOVE",        (0,   220, 170)),
    "LEFT_CLICK" : ("LEFT CLICK",  (80,  220,  80)),
    "RIGHT_CLICK": ("RIGHT CLICK", (120, 160, 255)),
    "SCROLL"     : ("SCROLL",      (255, 210,  50)),
    "VOLUME"     : ("VOLUME",      (255, 150,  50)),
    "BRIGHTNESS" : ("BRIGHTNESS",  (255, 230,  80)),
    "KB_TOGGLE"  : ("KEYBOARD",    (180, 120, 255)),
    "PAUSE"      : ("PAUSE",       (120, 120, 130)),
    "NO_HAND"    : ("NO HAND",     (70,   70,  80)),
}


def draw_hud(frame, gesture, fps, vol, bright, kb_open):
    label, color = GESTURE_INFO.get(gesture, ("–", (150,150,150)))
    ov = frame.copy()
    cv2.rectangle(ov, (8, 8), (240, 105), (8, 10, 16), -1)
    cv2.addWeighted(ov, 0.62, frame, 0.38, 0, frame)
    cv2.rectangle(frame, (8, 8), (240, 105), (50, 55, 70), 1)

    cv2.circle(frame, (25, 30), 7, color, -1)
    cv2.putText(frame, label, (38, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:4.0f}   VOL {vol:3.0f}%   BRI {bright:3.0f}%",
                (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,165,180), 1, cv2.LINE_AA)
    kb_c = (160, 100, 255) if kb_open else (60, 65, 80)
    cv2.putText(frame, f"KB {'ON ' if kb_open else 'OFF'}   [Q] quit",
                (14, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.38, kb_c, 1, cv2.LINE_AA)


def draw_vert_bar(frame, pct, x, y, bw, bh, color, label):
    ov = frame.copy()
    cv2.rectangle(ov, (x, y), (x+bw, y+bh), (8,10,16), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x, y), (x+bw, y+bh), (50,55,70), 1)
    filled = int(pct/100 * bh)
    cv2.rectangle(frame, (x, y+bh-filled), (x+bw, y+bh), color, -1)
    cv2.putText(frame, label,         (x, y-8),      cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{pct:.0f}%", (x, y+bh+16),  cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    smoother       = SmoothCursor()
    last_click     = 0
    scroll_ref_y   = None
    vol_ref_y      = None
    bright_ref_y   = None
    prev_t         = time.time()
    cursor_pos     = (0, 0)
    gesture        = "NO_HAND"

    cur_vol    = 50.0
    cur_bright = 50.0
    if VOLUME_OK:
        cur_vol = float(np.interp(volume_obj.GetMasterVolumeLevel(),
                                  [VOL_MIN, VOL_MAX], [0, 100]))
    if BRIGHT_OK:
        try: cur_bright = float(sbc.get_brightness()[0])
        except: pass

    kb_open        = False
    typed_text     = ""
    hovered_key    = None
    last_typed     = 0
    key_rects      = []
    panel_rect     = (0, 0, 0, 0)
    kb_key_w       = 60
    kb_key_h       = 50
    pinch_typed    = False
    last_kb_toggle = 0

    print("✅ Virtual Hand Control v3.1 — show your hand to the camera!")
    print("   Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera not accessible.")
            break

        frame  = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res    = hands_model.process(rgb)

        pts     = get_landmarks(frame, res)
        up      = fingers_up(pts)
        now     = time.time()
        gesture = detect_gesture(up, pts)

        # Hand skeleton
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        # ── KB TOGGLE ────────────────────────────────────────────
        if gesture == "KB_TOGGLE" and (now - last_kb_toggle) > 1.0:
            kb_open        = not kb_open
            last_kb_toggle = now
            if kb_open:
                key_rects, panel_rect, kb_key_w, kb_key_h = build_key_rects(fw, fh)
            else:
                key_rects = []

        # ── VOLUME ───────────────────────────────────────────────
        if gesture == "VOLUME" and pts:
            tip = pts[8]
            if vol_ref_y is None:
                vol_ref_y = tip[1]
            else:
                d = vol_ref_y - tip[1]
                if abs(d) > 4:
                    cur_vol   = float(np.clip(cur_vol + d * VOL_SENS, 0, 100))
                    vol_ref_y = tip[1]
                    if VOLUME_OK:
                        volume_obj.SetMasterVolumeLevel(
                            float(np.interp(cur_vol, [0,100], [VOL_MIN, VOL_MAX])), None)
            draw_vert_bar(frame, cur_vol, fw-72, 114, 26, 180, (255,150,50), "VOL")
        else:
            vol_ref_y = None

        # ── BRIGHTNESS ───────────────────────────────────────────
        if gesture == "BRIGHTNESS" and pts:
            tip = pts[8]
            if bright_ref_y is None:
                bright_ref_y = tip[1]
            else:
                d = bright_ref_y - tip[1]
                if abs(d) > 4:
                    cur_bright   = float(np.clip(cur_bright + d * BRIGHT_SENS, 0, 100))
                    bright_ref_y = tip[1]
                    if BRIGHT_OK:
                        try: sbc.set_brightness(int(cur_bright))
                        except: pass
            draw_vert_bar(frame, cur_bright, fw-112, 114, 26, 180, (255,230,80), "BRI")
        else:
            bright_ref_y = None

        # ── SCROLL ───────────────────────────────────────────────
        if gesture == "SCROLL" and pts:
            tip = pts[8]
            if scroll_ref_y is None:
                scroll_ref_y = tip[1]
            else:
                d = scroll_ref_y - tip[1]
                if abs(d) > 8:
                    pyautogui.scroll(int(d / SCROLL_SPEED))
                    scroll_ref_y = tip[1]
        else:
            scroll_ref_y = None

        # ── MOUSE ────────────────────────────────────────────────
        if not kb_open and pts:
            itip = pts[8]
            ttip = pts[4]

            if gesture == "MOVE":
                rx, ry     = map_to_screen(itip[0], itip[1], fw, fh)
                sx, sy     = smoother.smooth(rx, ry)
                cursor_pos = (sx, sy)
                pyautogui.moveTo(sx, sy)
                cv2.circle(frame, itip, 12, (255, 255, 255), -1)
                cv2.circle(frame, itip,  5, (0,  200, 160),  -1)

            elif gesture == "LEFT_CLICK" and (now - last_click) > CLICK_COOLDOWN:
                last_click = now
                pyautogui.click()
                cv2.circle(frame, itip, 22, (80,  220,  80),  2)
                cv2.circle(frame, itip,  8, (80,  220,  80), -1)
                cv2.putText(frame, "CLICK", (itip[0]+16, itip[1]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80,220,80), 2, cv2.LINE_AA)

            elif gesture == "RIGHT_CLICK" and (now - last_click) > CLICK_COOLDOWN:
                last_click = now
                pyautogui.rightClick()
                cv2.circle(frame, itip, 22, (120, 160, 255),  2)
                cv2.circle(frame, itip,  8, (120, 160, 255), -1)
                cv2.putText(frame, "R-CLICK", (itip[0]+16, itip[1]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (120,160,255), 2, cv2.LINE_AA)

        # ── KEYBOARD ─────────────────────────────────────────────
        if kb_open:
            draw_keyboard(frame, key_rects, panel_rect, hovered_key, typed_text)
            if pts:
                itip = pts[8]
                ttip = pts[4]
                cv2.circle(frame, itip, 12, (255, 255, 255), -1)
                cv2.circle(frame, itip,  5, (0,  200, 160),  -1)
                hk = get_hovered_key(itip, key_rects)
                if hk:
                    if hk != hovered_key:
                        hovered_key = hk
                    pd = dist(itip, ttip)
                    if pd < PINCH_KB_DIST and not pinch_typed and (now - last_typed) > KEY_COOLDOWN:
                        typed_text  = press_key(hovered_key, typed_text)
                        last_typed  = now
                        pinch_typed = True
                        cv2.circle(frame, itip, 26, (0, 220, 140), 3)
                    elif pd >= PINCH_KB_DIST:
                        pinch_typed = False
                else:
                    hovered_key = None
                    pinch_typed = False

        # ── HUD ──────────────────────────────────────────────────
        fps    = 1.0 / max(now - prev_t, 1e-9)
        prev_t = now
        draw_hud(frame, gesture, fps, cur_vol, cur_bright, kb_open)

        cv2.imshow("Virtual Hand Control  |  Q = Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Bye!")


if __name__ == "__main__":
    main()
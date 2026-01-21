import sys
import cv2
import numpy as np
import pygame
from pygame import Rect
import os
import traceback

# file dialog
import tkinter as tk
from tkinter import filedialog

# Import your detector class (assume detector.py implements RatDetector as before)
from detector import RatDetector

# --- Config (modifică după necesitate) ---
# Default model path left empty; user will load via button
DEFAULT_MODEL_PATH = None
MODEL_INPUT_SIZE = 256
MODEL_THRESHOLD = 0.95
DETECT_EVERY_N = 3   # rulează detect la fiecare 3 frame-uri (ajustează după performanță)

# Config UI
CAMERA_INDEX = 0
FPS = 30
BUTTON_PADDING = 10
BUTTON_W = 120
BUTTON_H = 36
BUTTON_COLOR_ON = (30, 180, 30)
BUTTON_COLOR_OFF = (180, 30, 30)
BUTTON_TEXT_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0)
TEXT_FONT_SIZE = 48
SMALL_FONT_SIZE = 20
BOX_COLOR = (255, 0, 0)
BOX_THICKNESS = 3

pygame.init()
clock = pygame.time.Clock()

# initial window size (resizable)
win_w, win_h = 1280, 720
screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
pygame.display.set_caption("Camera Toggle + Detector (Load Model Button)")

font = pygame.font.SysFont(None, TEXT_FONT_SIZE)
small_font = pygame.font.SysFont(None, SMALL_FONT_SIZE)

# UI state
camera_on = False
fullscreen = False
stretch_mode = False
camera = None

# detector state
detector = None
detection_result = None
frame_counter = 0
model_path = DEFAULT_MODEL_PATH
model_loaded = False
model_status_msg = "Model: none"

# Helper: open camera
def open_camera():
    cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cam.isOpened():
        print("Nu s-a putut deschide camera.")
        return None
    return cam

def close_camera():
    global camera
    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass
        camera = None

def compute_letterbox_params(src_w, src_h, dest_w, dest_h):
    scale = min(dest_w / src_w, dest_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    offset_x = (dest_w - new_w) // 2
    offset_y = (dest_h - new_h) // 2
    return scale, new_w, new_h, offset_x, offset_y

def letterbox_blit(surf, target_surf, dest_rect, keep_aspect=True):
    dw, dh = dest_rect[2], dest_rect[3]
    sw, sh = surf.get_width(), surf.get_height()
    if keep_aspect:
        scale, new_w, new_h, ox, oy = compute_letterbox_params(sw, sh, dw, dh)
        scaled = pygame.transform.smoothscale(surf, (new_w, new_h))
        target_surf.fill(BG_COLOR, dest_rect)
        target_surf.blit(scaled, (dest_rect[0] + ox, dest_rect[1] + oy))
        return {"scale": scale, "ox": dest_rect[0] + ox, "oy": dest_rect[1] + oy, "draw_w": new_w, "draw_h": new_h}
    else:
        scaled = pygame.transform.smoothscale(surf, (dw, dh))
        target_surf.blit(scaled, (dest_rect[0], dest_rect[1]))
        return {"scale": dw / sw, "ox": dest_rect[0], "oy": dest_rect[1], "draw_w": dw, "draw_h": dh}

def draw_button(surface, text, rect, is_on):
    color = BUTTON_COLOR_ON if is_on else BUTTON_COLOR_OFF
    pygame.draw.rect(surface, color, rect, border_radius=6)
    pygame.draw.rect(surface, (255,255,255), rect, 2, border_radius=6)
    txt = small_font.render(text, True, BUTTON_TEXT_COLOR)
    tx = rect.x + (rect.w - txt.get_width()) // 2
    ty = rect.y + (rect.h - txt.get_height()) // 2
    surface.blit(txt, (tx, ty))

def point_in_rect(pt, rect):
    return rect.left <= pt[0] <= rect.right and rect.top <= pt[1] <= rect.bottom

def draw_camera_closed(screen):
    screen.fill(BG_COLOR)
    text = font.render("Camera închisă", True, (255,255,255))
    screen.blit(text, ((screen.get_width()-text.get_width())//2, (screen.get_height()-text.get_height())//2))

# File dialog loader (synchronous). Use Tkinter dialog to pick model file.
def load_model_via_dialog():
    global detector, model_path, model_loaded, model_status_msg
    try:
        # Init hidden root for filedialog
        root = tk.Tk()
        root.withdraw()
        filetypes = [
            ("PyTorch model (.pth)", "*.pth"),
            ("Keras model (.h5)", "*.h5"),
            ("All files", "*.*"),
        ]
        filename = filedialog.askopenfilename(title="Select model file", filetypes=filetypes)
        root.destroy()
    except Exception as e:
        print("File dialog error:", e)
        model_status_msg = f"Model dialog error"
        return

    if not filename:
        # user cancelled
        return

    # Try to instantiate detector
    try:
        model_status_msg = f"Loading: {os.path.basename(filename)}..."
        pygame.display.set_caption(f"Loading model: {os.path.basename(filename)}")
        # instantiate; RatDetector should accept path, input_size and threshold
        det = RatDetector(filename, input_size=MODEL_INPUT_SIZE, class_threshold=MODEL_THRESHOLD)
        detector = det
        model_path = filename
        model_loaded = True
        model_status_msg = f"Model loaded: {os.path.basename(filename)}"
        print("Model loaded from:", filename)
    except Exception as e:
        print("Failed loading model:", e)
        traceback.print_exc()
        detector = None
        model_loaded = False
        model_path = None
        model_status_msg = f"Failed to load model: {os.path.basename(filename)}"

def unload_model():
    global detector, model_path, model_loaded, model_status_msg
    detector = None
    model_path = None
    model_loaded = False
    model_status_msg = "Model: none"
    print("Model unloaded.")

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_f:
                fullscreen = not fullscreen
                if fullscreen:
                    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                else:
                    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
            elif event.key == pygame.K_t:
                stretch_mode = not stretch_mode

        elif event.type == pygame.VIDEORESIZE:
            win_w, win_h = event.w, event.h
            if not fullscreen:
                screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            sw, sh = screen.get_size()

            # top-right camera toggle button
            top_btn_rect = Rect(sw - BUTTON_W - BUTTON_PADDING, BUTTON_PADDING, BUTTON_W, BUTTON_H)
            if point_in_rect((mx,my), top_btn_rect):
                camera_on = not camera_on
                if camera_on:
                    camera = open_camera()
                    if camera is None:
                        camera_on = False
                    else:
                        detection_result = None
                else:
                    close_camera()
                    detection_result = None
                continue

            # bottom-right load/unload model button
            bottom_btn_rect = Rect(sw - BUTTON_W - BUTTON_PADDING, sh - BUTTON_H - BUTTON_PADDING, BUTTON_W, BUTTON_H)
            if point_in_rect((mx,my), bottom_btn_rect):
                # If a model is loaded, unload. Otherwise open dialog to load.
                if model_loaded:
                    unload_model()
                else:
                    load_model_via_dialog()
                continue

    sw, sh = screen.get_size()
    top_btn_rect = Rect(sw - BUTTON_W - BUTTON_PADDING, BUTTON_PADDING, BUTTON_W, BUTTON_H)
    bottom_btn_rect = Rect(sw - BUTTON_W - BUTTON_PADDING, sh - BUTTON_H - BUTTON_PADDING, BUTTON_W, BUTTON_H)

    if camera_on and camera is not None:
        ret, frame = camera.read()
        if not ret:
            close_camera()
            camera_on = False
            draw_camera_closed(screen)
        else:
            # frame: BGR numpy
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Create pygame surface from frame
            try:
                surf = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
            except Exception:
                frame_rgb = np.rot90(frame_rgb)
                surf = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")

            dest_rect = (0, 0, sw, sh)
            letterbox_info = letterbox_blit(surf, screen, dest_rect, keep_aspect=(not stretch_mode))

            # detect at an interval only if model is loaded
            frame_counter += 1
            if model_loaded and detector is not None and (frame_counter % DETECT_EVERY_N == 0):
                try:
                    det = detector.predict(frame)
                    detection_result = det  # either None or dict
                except Exception as e:
                    print("Detection error:", e)
                    detection_result = None

            # if we have bbox => map to screen coords
            if detection_result is not None:
                prob = detection_result["prob"]
                bx, by, bw, bh = detection_result["bbox"]  # in pixels (frame coords)
                # map using letterbox_info
                scale = letterbox_info["scale"]
                ox = letterbox_info["ox"]
                oy = letterbox_info["oy"]

                # scaled top-left:
                sx = int(round(ox + bx * scale))
                sy = int(round(oy + by * scale))
                sw_box = int(round(bw * scale))
                sh_box = int(round(bh * scale))

                # draw rectangle and label
                pygame.draw.rect(screen, BOX_COLOR, (sx, sy, sw_box, sh_box), BOX_THICKNESS)
                label = f"sobolan {prob:.2f}"
                lbl_surf = small_font.render(label, True, (255,255,255))
                # background for label
                lbl_bg = pygame.Surface((lbl_surf.get_width()+6, lbl_surf.get_height()+4))
                lbl_bg.fill((0,0,0))
                lbl_bg.set_alpha(180)
                screen.blit(lbl_bg, (sx, max(0, sy - lbl_surf.get_height() - 6)))
                screen.blit(lbl_surf, (sx+3, max(0, sy - lbl_surf.get_height() - 4)))

    else:
        draw_camera_closed(screen)

    # Draw the camera toggle button (top-right)
    draw_button(screen, "Oprește" if camera_on else "Pornește", top_btn_rect, camera_on)

    # Draw the model load/unload button (bottom-right)
    draw_button(screen, "Unload" if model_loaded else "Load model", bottom_btn_rect, model_loaded)

    # small help text
    help_text = small_font.render("F: fullscreen | T: stretch video | Esc: ieșire", True, (200,200,200))
    screen.blit(help_text, (10, sh - help_text.get_height() - 10))

    # model status (left-bottom)
    status_surf = small_font.render(model_status_msg, True, (220,220,220))
    screen.blit(status_surf, (10, 10))

    pygame.display.flip()
    clock.tick(FPS)

# Cleanup
close_camera()
pygame.quit()
sys.exit()

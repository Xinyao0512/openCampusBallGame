# オープンキャンパス デモ

import cv2
import pygame
import sys
import random
import mediapipe as mp

# カメラの初期化
cap = cv2.VideoCapture(1)

# MediaPipeの初期化
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

# Pygameの初期化
pygame.init()
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("バスケットボール リフティング ゲーム")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

# ゲーム変数
BALL_RADIUS = 30
gravity = 0.5
bounce_strength = -20
angle = 0

# 画像リソースをロードする
try:
    basketball_img = pygame.image.load("img/basketball.png").convert_alpha()
    basketball_img = pygame.transform.scale(basketball_img, (BALL_RADIUS * 2, BALL_RADIUS * 2))
    ikun_img = pygame.image.load("img/iKUN.png").convert_alpha()
except:
    print("resource not found")
    pygame.quit()
    cap.release()
    sys.exit()

# 状態の初期化
high_score = 0

def reset_game():
    global ball_pos, ball_vel, score, game_over, angle
    ball_pos = [WIDTH // 2, HEIGHT // 4]
    ball_vel = [random.choice([-3, 3]), 0]
    score = 0
    game_over = False
    angle = 0

reset_game()

def draw_basketball(surface, center, rotation_angle):
    rotated_ball = pygame.transform.rotate(basketball_img, rotation_angle)
    rect = rotated_ball.get_rect(center=(int(center[0]), int(center[1])))
    surface.blit(rotated_ball, rect.topleft)

def draw_text(text, color, y):
    render = font.render(text, True, color)
    screen.blit(render, (WIDTH // 2 - render.get_width() // 2, y))

def draw_score(score, high_score):
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    high_text = font.render(f"High Score: {high_score}", True, (255, 215, 0))
    screen.blit(score_text, (10, 10))
    screen.blit(high_text, (10, 40))

f1_pressed = False

while True:
    ret, frame = cap.read()
    if not ret:
        cap.release()
        pygame.quit()
        sys.exit()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    face_results = face_detection.process(rgb_frame)

    # カメラの背景を表示
    screen.fill((0, 0, 0))
    cam_surface = pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cam_surface = pygame.transform.rotate(cam_surface, -90)
    cam_surface = pygame.transform.scale(cam_surface, (WIDTH, HEIGHT))
    screen.blit(cam_surface, (0, 0))

    # 顔を検出して画像を貼り付ける
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int((1 - bbox.xmin - bbox.width) * WIDTH)
            y = int(bbox.ymin * HEIGHT)
            w = int(bbox.width * WIDTH)
            h = int(bbox.height * HEIGHT)

            scaled_ikun = pygame.transform.scale(ikun_img, (w, h))
            screen.blit(scaled_ikun, (x, y))

    # ボールの動き
    if not game_over:
        ball_vel[1] += gravity
        ball_pos[0] += ball_vel[0]
        ball_pos[1] += ball_vel[1]
        angle = (angle + 5) % 360

        if ball_pos[0] <= BALL_RADIUS or ball_pos[0] >= WIDTH - BALL_RADIUS:
            ball_vel[0] = -ball_vel[0]
        if ball_pos[1] >= HEIGHT - BALL_RADIUS:
            game_over = True
            ball_vel = [0, 0]

    if not game_over:
        draw_basketball(screen, ball_pos, angle)

    # 手の認識
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            x_coords = [1 - lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * WIDTH)
            x_max = int(max(x_coords) * WIDTH)
            y_min = int(min(y_coords) * HEIGHT)
            y_max = int(max(y_coords) * HEIGHT)

            hand_rect = pygame.Rect(x_min, y_min, x_max - x_min, y_max - y_min)

            ball_rect = pygame.Rect(ball_pos[0] - BALL_RADIUS, ball_pos[1] - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)
            if not game_over and hand_rect.colliderect(ball_rect):
                ball_vel[1] = bounce_strength
                ball_vel[0] += random.choice([-1, 0, 1])
                score += 1
                if score > high_score:
                    high_score = score

    draw_score(score, high_score)

    if game_over:
        draw_text("Game Over", (255, 0, 0), HEIGHT // 2 - 20)
        draw_text("Press F1 to restart", (255, 255, 255), HEIGHT // 2 + 20)
    else:
        draw_text("Press F1 to restart", (180, 180, 180), HEIGHT - 30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F1 and not f1_pressed:
                reset_game()
                f1_pressed = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_F1:
                f1_pressed = False

    pygame.display.flip()
    clock.tick(60)

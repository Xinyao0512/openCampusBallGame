# 電気通信大学 高橋裕樹 研究室 オープンキャンパス デモ
# v1.0.6_手球

import cv2
import pygame
import sys
import random
import os
import mediapipe as mp

# カメラの初期化
cap = cv2.VideoCapture(1)

# MediaPipe の初期化
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Pygame の初期化
pygame.init()
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ボール ゲーム")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

# リソースの読み込み
try:
    basketball_img = pygame.image.load("img/ball.png").convert_alpha()
    basketball_img = pygame.transform.scale(basketball_img, (60, 60))
except:
    print("resource not found")
    pygame.quit()
    cap.release()
    sys.exit()

# ハイスコアの読み込みと保存の関数
def load_high_score():
    if os.path.exists("highscore.dat"):
        with open("highscore.dat", "r") as f:
            try:
                return int(f.read())
            except:
                return 0
    return 0

def save_high_score(score):
    with open("highscore.dat", "w") as f:
        f.write(str(score))

high_score = load_high_score()

# 顔の幅から距離を推定する関数
def estimate_distance_from_face_width(face_width_px):
    real_face_width_mm = 160
    focal_length_px = 600
    if face_width_px == 0:
        return None
    distance_mm = (real_face_width_mm * focal_length_px) / face_width_px
    return distance_mm / 1000

# ゲームの初期化
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

# メインループ
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    face_results = face_detection.process(rgb_frame)
    face_mesh_results = face_mesh.process(rgb_frame)

    screen.fill((0, 0, 0))
    cam_surface = pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cam_surface = pygame.transform.rotate(cam_surface, -90)
    cam_surface = pygame.transform.scale(cam_surface, (WIDTH, HEIGHT))
    screen.blit(cam_surface, (0, 0))

    distance_ok = False

    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int((1 - bbox.xmin - bbox.width) * WIDTH)
            y = int(bbox.ymin * HEIGHT)
            w = int(bbox.width * WIDTH)

            est_distance = estimate_distance_from_face_width(w)
            if est_distance:
                dist_text = f"{est_distance:.2f}m"
                if est_distance < 0.8:
                    dist_text += "  Too Close"
                    color = (255, 0, 0)
                # elif est_distance > 1.7:
                #     dist_text += "  Too Far"
                #     color = (255, 0, 0)
                else:
                    dist_text += "  OK"
                    color = (0, 255, 0)
                    distance_ok = True

                dist_render = font.render(dist_text, True, color)
                screen.blit(dist_render, (x, y - 30))

    # 🔹 FaceMesh 表示（镜像X轴）
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                x = int((1 - lm.x) * WIDTH)
                y = int(lm.y * HEIGHT)
                pygame.draw.circle(screen, (0, 255, 0), (x, y), 1)

    if not game_over and distance_ok:
        ball_vel[1] += 0.5
        ball_pos[0] += ball_vel[0]
        ball_pos[1] += ball_vel[1]
        angle = (angle + 5) % 360

        if ball_pos[0] <= 30 or ball_pos[0] >= WIDTH - 30:
            ball_vel[0] = -ball_vel[0]
        if ball_pos[1] >= HEIGHT - 30:
            game_over = True
            ball_vel = [0, 0]

    if not game_over:
        draw_basketball(screen, ball_pos, angle)

    if hand_results.multi_hand_landmarks and not game_over and distance_ok:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            x_coords = [(1 - lm.x) * WIDTH for lm in hand_landmarks.landmark]
            y_coords = [lm.y * HEIGHT for lm in hand_landmarks.landmark]

            # 🔹 绘制点和线
            landmark_points = []
            for lm in hand_landmarks.landmark:
                x = int((1 - lm.x) * WIDTH)
                y = int(lm.y * HEIGHT)
                landmark_points.append((x, y))
                pygame.draw.circle(screen, (0, 0, 255), (x, y), 3)

            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                    pygame.draw.line(screen, (0, 255, 255), landmark_points[start_idx], landmark_points[end_idx], 2)

            # 🔹 碰撞检测
            x_min = int(min(x_coords))
            x_max = int(max(x_coords))
            y_min = int(min(y_coords))
            y_max = int(max(y_coords))
            hand_rect = pygame.Rect(x_min, y_min, x_max - x_min, y_max - y_min)
            ball_rect = pygame.Rect(ball_pos[0] - 30, ball_pos[1] - 30, 60, 60)
            if hand_rect.colliderect(ball_rect):
                ball_vel[1] = -20
                ball_vel[0] += random.choice([-1, 0, 1])
                score += 1
                if score > high_score:
                    high_score = score
                    save_high_score(high_score)

    draw_score(score, high_score)

    if game_over:
        draw_text("Game Over", (255, 0, 0), HEIGHT // 2 - 20)
        draw_text("Press F1 to restart/F2 to reset", (180, 180, 180), HEIGHT - 30)
    elif not distance_ok:
        draw_text("Please keep 0.8~1.7m to start", (255, 255, 0), HEIGHT // 2 + 60)
        draw_text("Game Paused", (255, 0, 0), HEIGHT // 2 + 100)
    else:
        draw_text("Press F1 to restart/F2 to reset", (180, 180, 180), HEIGHT - 30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            hands.close()
            face_detection.close()
            face_mesh.close()
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F1:
                reset_game()
            elif event.key == pygame.K_F2:
                high_score = 0
                save_high_score(0)

    pygame.display.flip()
    clock.tick(60)

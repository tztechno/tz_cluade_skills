import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # --- Part 1: 車両検知 (Detection & Extraction) の設定 ---
    # YOLO11モデルのロード（高速なnanoモデルを使用）
    model = YOLO('yolo11n.pt') 
    
    # 入力動画の設定（mp4ファイル）
    video_path = 'traffic_video.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # 動画のFPSを取得し、1フレームあたりの時間を算出
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0  # 取得できない場合のデフォルト
    time_delta = 1 / fps
    
    # --- Part 2: 速度推定ロジック (Speed Calculation) の定数 ---
    REAL_WIDTH = 1.8  # 一般的な車両の幅(m)
    
    # トラッキング用：{track_id: 前フレームのx中心座標}
    prev_positions = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Part 1: 検知とトラッキング
        # persist=True でID追跡を有効化。車両クラス(2,3,5,7)のみを対象
        results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)

        if results[0].boxes.id is not None:
            # 座標(xywh)とトラッキングIDを取得
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x_center, y_center, width, height = box

                # --- Part 2: 速度推定ロジックの実行 ---
                if track_id in prev_positions:
                    prev_x = prev_positions[track_id]
                    
                    # 1. ピクセル移動量 (ΔPixel)
                    delta_pixel = abs(x_center - prev_x)
                    
                    # 2. 速度算出（設計式の適用）
                    # Velocity (m/s) = (ΔPixel * 実車幅) / (画面上の車幅 * ΔTime)
                    velocity_mps = (delta_pixel * REAL_WIDTH) / (width * time_delta)
                    speed_kmh = velocity_mps * 3.6  # m/s -> km/h 変換
                    
                    # 3. 画面への表示
                    label = f"ID:{track_id} {speed_kmh:.1f} km/h"
                    cv2.putText(frame, label, (int(x_center - width/2), int(y_center - height/2 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # 矩形の描画
                    cv2.rectangle(frame, (int(x_center-width/2), int(y_center-height/2)), 
                                  (int(x_center+width/2), int(y_center+height/2)), (0, 255, 0), 2)
                
                # 位置情報の更新
                prev_positions[track_id] = x_center

        # 結果表示
        cv2.imshow("Speed Estimation Simulation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
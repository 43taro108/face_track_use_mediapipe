# -*- coding: utf-8 -*-
"""
Created on Sun May 18 12:16:11 2025

@author: ktrpt
"""
import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 0. Page config
st.set_page_config(page_title="Single-Frame Face Landmarks", page_icon="🖼️", layout="wide")
st.title("🖼️ 1フレーム顔ランドマーク抽出＆3Dプロット")

# Sidebar settings
st.sidebar.header("🔧 設定")
max_faces = st.sidebar.number_input("同時検出する顔の数", min_value=1, max_value=4, value=1)
min_det_conf = st.sidebar.slider("検出信頼度の閾値", min_value=0.1, max_value=1.0, value=0.5)
min_track_conf = st.sidebar.slider("追跡信頼度の閾値", min_value=0.1, max_value=1.0, value=0.5)

# File uploader: video or image
media_file = st.file_uploader("動画(mp4/avi/mov)または画像(jpg/png)をアップロード", type=["mp4","avi","mov","jpg","jpeg","png"])

if media_file:
    # Save upload to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(media_file.read())
    path = tfile.name

    # Determine media type
    is_video = path.lower().endswith(('.mp4', '.avi', '.mov'))

    if is_video:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            st.error("動画ファイルを開けませんでした。")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Frame selection slider
            frame_no = st.slider("抽出するフレーム番号", min_value=0, max_value=total_frames-1, value=0)

            # Preview frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            cap.release()
            if ret:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_no}", use_column_width=True)
                selected_frame = frame
            else:
                st.error("プレビュー用フレームを読み込めませんでした。")
                st.stop()
    else:
        # Image
        img = cv2.imdecode(np.frombuffer(open(path, 'rb').read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="アップロード画像", use_column_width=True)
        selected_frame = img

    # Extraction button
    if st.button("▶ ランドマーク抽出＆CSV出力"):
        st.info("処理中…")
        # MediaPipe FaceMesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=int(max_faces),
            refine_landmarks=False,
            min_detection_confidence=float(min_det_conf),
            min_tracking_confidence=float(min_track_conf)
        )
        # Process one frame
        rgb = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        face_mesh.close()

        if not results.multi_face_landmarks:
            st.warning("顔が検出されませんでした。")
        else:
            records = []
            h, w, _ = selected_frame.shape
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                for lm_id, lm in enumerate(face_landmarks.landmark):
                    x_px = lm.x * w
                    y_px = lm.y * h
                    z_rel = lm.z * max(w, h)
                    records.append({
                        "face_id": face_id,
                        "landmark_id": lm_id,
                        "x_px": x_px,
                        "y_px": y_px,
                        "z_rel": z_rel
                    })
            df = pd.DataFrame(records)
            st.success(f"抽出完了！ 顔{len(results.multi_face_landmarks)}件, ランドマーク数: {len(df)}行")

            # Show as table
            st.dataframe(df, use_container_width=True)

            # CSV download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 CSV をダウンロード", data=csv,
                file_name="face_landmarks.csv", mime="text/csv"
            )

            # 3D plot
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df['x_px'], df['y_px'], df['z_rel'], s=10)
            ax.set_xlabel('X (px)')
            ax.set_ylabel('Y (px)')
            ax.set_zlabel('Z (relative)')
            ax.set_title('Face Landmarks 3D Scatter')
            ax.invert_yaxis()
            st.pyplot(fig)

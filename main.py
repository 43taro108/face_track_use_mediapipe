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
from tqdm import tqdm

# 0. Page config
st.set_page_config(page_title="Video Face Landmark 3D", page_icon="📹", layout="wide")
st.title("📹 動画から顔ランドマーク抽出＆3Dプロット")

# Sidebar settings
st.sidebar.header("🔧 設定")
frame_step = st.sidebar.number_input("何フレームごとに抽出", min_value=1, max_value=100, value=5)
max_faces = st.sidebar.number_input("同時検出する顔の数", min_value=1, max_value=4, value=1)
min_det_conf = st.sidebar.slider("検出信頼度の閾値", min_value=0.1, max_value=1.0, value=0.5)
min_track_conf = st.sidebar.slider("追跡信頼度の閾値", min_value=0.1, max_value=1.0, value=0.5)

# File uploader
video_file = st.file_uploader("解析したい動画ファイルをアップロード", type=["mp4","avi","mov"])

if video_file:
    # Save upload to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # Preview frame via slider
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("動画ファイルを開けませんでした。")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_preview = st.slider("プレビュー用フレーム選択", min_value=0, max_value=total_frames-1, value=0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_preview)
        ret, frame = cap.read()
        if ret:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_preview}", use_column_width=True)
    cap.release()

    # Extraction button
    if st.button("▶ 抽出開始"):
        st.info("処理中…しばらくお待ちください")
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # MediaPipe FaceMesh setup
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=int(max_faces),
            refine_landmarks=False,
            min_detection_confidence=float(min_det_conf),
            min_tracking_confidence=float(min_track_conf)
        )

        records = []
        progress_bar = st.progress(0)
        frame_indices = list(range(0, total_frames, int(frame_step)))

        for idx, frame_no in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                progress_bar.progress((idx + 1) / len(frame_indices))
                continue
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                progress_bar.progress((idx + 1) / len(frame_indices))
                continue
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                for lm_id, lm in enumerate(face_landmarks.landmark):
                    x_px = lm.x * width
                    y_px = lm.y * height
                    z_rel = lm.z * max(width, height)
                    records.append({
                        "frame": frame_no,
                        "face_id": face_id,
                        "landmark_id": lm_id,
                        "x_px": x_px,
                        "y_px": y_px,
                        "z_rel": z_rel
                    })
            progress_bar.progress((idx + 1) / len(frame_indices))

        cap.release()
        face_mesh.close()

        if not records:
            st.warning("顔が検出されませんでした。設定を調整してください。")
        else:
            df = pd.DataFrame(records)
            st.success(f"抽出完了！ 行数: {len(df)}")

            # CSV download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 CSV をダウンロード", data=csv,
                file_name="face_landmarks.csv", mime="text/csv"
            )

            # 3D plot selection
            sel_frame = st.sidebar.selectbox("3D表示するフレーム", sorted(df['frame'].unique()))
            df0 = df[df['frame'] == sel_frame]

            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df0['x_px'], df0['y_px'], df0['z_rel'], s=10)
            ax.set_xlabel('X (px)')
            ax.set_ylabel('Y (px)')
            ax.set_zlabel('Z (relative)')
            ax.set_title(f'Face Landmarks @ frame {sel_frame}')
            ax.invert_yaxis()
            st.pyplot(fig)

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
st.set_page_config(page_title="Face Landmark Mesh 3D", page_icon="🖼️", layout="wide")
st.title("🖼️ 1フレーム顔ランドマークメッシュ＆3D正面表示 (ロール＆ヨー補正)")

# Sidebar settings
st.sidebar.header("🔧 設定")
max_faces = st.sidebar.number_input("同時検出する顔の数", min_value=1, max_value=4, value=1)
min_det_conf = st.sidebar.slider("検出信頼度の閾値", min_value=0.1, max_value=1.0, value=0.5)
min_track_conf = st.sidebar.slider("追跡信頼度の閾値", min_value=0.1, max_value=1.0, value=0.5)

# File uploader: video or image
media_file = st.file_uploader(
    "動画(mp4/avi/mov)または画像(jpg/png)をアップロード",
    type=["mp4","avi","mov","jpg","jpeg","png"]
)

# Utility: equal aspect ratio for 3D axes

def set_axes_equal(ax):
    extents = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    centers = np.mean(extents, axis=1)
    max_range = np.max(extents[:,1] - extents[:,0]) / 2
    for ctr, axis in zip(centers, 'xyz'):
        getattr(ax, f'set_{axis}lim')(ctr - max_range, ctr + max_range)

if media_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(media_file.read())
    path = tfile.name
    is_video = path.lower().endswith(('.mp4', '.avi', '.mov'))

    if is_video:
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_no = st.slider("抽出するフレーム番号", 0, total_frames-1, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            st.error("プレビュー用フレームを読み込めませんでした。")
            st.stop()
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_no}", use_column_width=True)
        selected_frame = frame
    else:
        img = cv2.imdecode(np.frombuffer(open(path,'rb').read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="アップロード画像", use_column_width=True)
        selected_frame = img

    if st.button("▶ ランドマーク抽出＆可視化"):
        st.info("処理中…")
        mp_face = mp.solutions.face_mesh
        face_mesh = mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=int(max_faces),
            refine_landmarks=False,
            min_detection_confidence=float(min_det_conf),
            min_tracking_confidence=float(min_track_conf)
        )
        rgb = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        face_mesh.close()

        if not results.multi_face_landmarks:
            st.warning("顔が検出されませんでした。")
        else:
            # Collect landmarks
            h, w, _ = selected_frame.shape
            records = []
            for face_landmarks in results.multi_face_landmarks:
                for lm_id, lm in enumerate(face_landmarks.landmark):
                    x = lm.x * w
                    y = lm.y * h
                    z = lm.z * max(w, h)
                    records.append([lm_id, x, y, z])
            df = pd.DataFrame(records, columns=["landmark_id","x","y","z"])

            # Roll correction using eyes (33,263)
            pL = df[df.landmark_id==33][['x','y']].values[0]
            pR = df[df.landmark_id==263][['x','y']].values[0]
            roll = np.arctan2(pR[1]-pL[1], pR[0]-pL[0])
            # Yaw correction using cheeks (234,454)
            pCL = df[df.landmark_id==234][['x','z']].values[0]
            pCR = df[df.landmark_id==454][['x','z']].values[0]
            yaw = np.arctan2(pCR[2]-pCL[2], pCR[0]-pCL[0])

            # Combined rotation: first roll around Z, then yaw around Y
            # Rotation matrices
            Rz = np.array([[ np.cos(-roll), -np.sin(-roll), 0],
                           [ np.sin(-roll),  np.cos(-roll), 0],
                           [            0,             0, 1]])
            Ry = np.array([[ np.cos(-yaw), 0, np.sin(-yaw)],
                           [           0, 1,            0],
                           [-np.sin(-yaw), 0, np.cos(-yaw)]])
            pts = df[['x','y','z']].values
            pts = pts.dot(Rz.T).dot(Ry.T)
            df[['x','y','z']] = pts

            st.success(f"抽出完了！ ランドマーク数: {len(df)}行")
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 CSV をダウンロード", csv, file_name="face_landmarks.csv", mime="text/csv")

            # 3D mesh frontal XY view
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, projection='3d')
            coords = {row.landmark_id:(row.x,row.y,row.z) for _,row in df.iterrows()}
            for start,end in mp_face.FACEMESH_TESSELATION:
                if start in coords and end in coords:
                    xs,ys,zs = zip(coords[start],coords[end])
                    ax.plot(xs,ys,zs,linewidth=0.5)
            set_axes_equal(ax)
            ax.view_init(elev=90,azim=0)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Face Landmark Mesh (Roll & Yaw Corrected)')
            st.pyplot(fig)

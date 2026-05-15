---
name: youtube-motion-detect-recorder
description: >
  Build a headless motion-detection recorder that monitors a YouTube livestream
  or video and automatically saves video clips whenever movement is detected.
  Use this skill whenever the user wants to: watch a YouTube stream for motion,
  record clips from a live camera feed, automate surveillance-style recording,
  detect activity in a video without a GUI, or run OpenCV motion detection in a
  Jupyter notebook or server environment. Trigger even if the user just says
  "monitor a YouTube video for movement" or "save clips when something happens
  on stream" — the headless approach and yt-dlp extraction are non-obvious and
  this skill captures the exact working pattern.
---

# YouTube Motion Detection Recorder

A headless Python script that:
1. Extracts a playable stream URL from any YouTube video/livestream using **yt-dlp**
2. Reads frames with **OpenCV + FFmpeg backend**
3. Compares consecutive frames to detect motion
4. Saves MP4 clips to disk whenever motion exceeds a configurable threshold
5. Logs progress to stdout — **no GUI window required**

---

## Dependencies

```bash
pip install yt-dlp opencv-python-headless numpy
```

> Use `opencv-python-headless` (not `opencv-python`) on servers/Jupyter to
> avoid the Qt/GTK dependency that causes kernel crashes.

---

## Critical Rules

| Rule | Why |
|------|-----|
| **Never call `cv2.imshow()`, `cv2.waitKey()`, or `cv2.destroyAllWindows()`** | These crash the kernel/process on any headless environment (Jupyter, Docker, SSH) |
| **Always pass `cv2.CAP_FFMPEG`** to `VideoCapture` | YouTube streams are HLS/DASH; OpenCV needs the FFmpeg backend to decode them |
| **Use `best[ext=mp4][height<=720]`** as the yt-dlp format selector | Avoids separate audio/video tracks that OpenCV cannot mux on the fly |
| **Re-extract the stream URL if you restart** | YouTube CDN URLs expire in ~6 hours |

---

## Core Algorithm: Motion Detection

Motion is detected by comparing two consecutive greyscale frames:

```
Frame N  ──┐
            ├─► Grayscale ──► GaussianBlur(21×21) ──┐
Frame N+1 ─┘                                         ├─► absdiff ──► threshold(25) ──► dilate ──► findContours
                                                      │
                                              sum area of contours > threshold?
                                                      │
                                              YES ──► start_recording()
```

**Key parameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `motion_threshold` | 5000 | Minimum total changed-pixel area (px²) to count as motion. Lower = more sensitive. |
| `record_duration` | 5 | Seconds to record after motion is detected |
| `watch_duration` | 300 | Total monitoring window in seconds |
| Blur kernel | `(21, 21)` | Larger = ignores fine noise; smaller = more sensitive to texture changes |
| Diff threshold | `25` | Per-pixel brightness change needed to count as "moved" |

---

## Complete Working Code

```python
import cv2
import numpy as np
import time
from datetime import datetime
import os
import yt_dlp


class MotionDetectRecorder:
    def __init__(self, watch_duration=300, record_duration=5, motion_threshold=5000):
        self.watch_duration = watch_duration
        self.record_duration = record_duration
        self.motion_threshold = motion_threshold
        self.is_recording = False
        self.record_end_time = 0
        self.video_writer = None
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)

    def get_youtube_stream_url(self, url):
        ydl_opts = {
            'format': 'best[ext=mp4][height<=720]/best[ext=mp4]/best',
            'quiet': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if 'url' in info:
                    return info['url']
                for f in reversed(info.get('formats', [])):
                    if f.get('url') and f.get('vcodec') != 'none':
                        return f['url']
        except Exception as e:
            print(f"yt-dlp error: {e}")
        return None

    def detect_motion(self, frame1, frame2):
        def preprocess(f):
            return cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        diff = cv2.absdiff(preprocess(frame1), preprocess(frame2))
        thresh = cv2.dilate(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1],
                            None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 500)
        return total_area > self.motion_threshold

    def start_recording(self, frame, fps=30.0):
        if not self.is_recording:
            self.is_recording = True
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.output_dir, f"motion_{ts}.mp4")
            h, w = frame.shape[:2]
            self.video_writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f"[{ts}] Recording → {path}")

    def stop_recording(self):
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Recording stopped")

    def process_youtube_stream(self, url):
        stream_url = self.get_youtube_stream_url(url)
        if not stream_url:
            print("ERROR: Could not extract stream URL")
            return

        # CAP_FFMPEG is required for HLS/DASH YouTube streams
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("ERROR: Stream could not be opened. Check FFmpeg support.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0

        ret, prev_frame = cap.read()
        if not ret:
            print("ERROR: Could not read first frame")
            cap.release()
            return

        start_time = time.time()
        frame_count = 0
        last_log = start_time

        print(f"Monitoring: {self.watch_duration}s | "
              f"Threshold: {self.motion_threshold}px² | "
              f"Clip length: {self.record_duration}s")

        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed >= self.watch_duration:
                    print("Monitoring window complete.")
                    break

                ret, current_frame = cap.read()
                if not ret:
                    print("Stream ended.")
                    break

                if self.detect_motion(prev_frame, current_frame) and not self.is_recording:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Motion detected!")
                    self.start_recording(current_frame, fps)
                    self.record_end_time = time.time() + self.record_duration

                if self.is_recording:
                    self.video_writer.write(current_frame)
                    if time.time() >= self.record_end_time:
                        self.stop_recording()

                prev_frame = current_frame
                frame_count += 1

                if time.time() - last_log >= 10:
                    remaining = max(0, self.watch_duration - elapsed)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Frames: {frame_count} | Remaining: {remaining:.0f}s")
                    last_log = time.time()

        except KeyboardInterrupt:
            print("Interrupted.")
        finally:
            if self.is_recording:
                self.stop_recording()
            cap.release()
            print(f"Done. {frame_count} frames. Clips in ./{self.output_dir}/")


# ── Usage ──────────────────────────────────────────────────────────────────────
detector = MotionDetectRecorder(
    watch_duration=300,    # how long to monitor (seconds)
    record_duration=5,     # clip length after motion (seconds)
    motion_threshold=1000  # sensitivity — lower = triggers more easily
)
detector.process_youtube_stream("https://www.youtube.com/watch?v=YOUR_VIDEO_ID")
```

---

## Tuning Guide

**Too many false triggers (wind, lighting changes):**
- Increase `motion_threshold` (e.g. 5000 → 10000)
- Increase blur kernel from `(21,21)` to `(31,31)`
- Increase per-pixel threshold from `25` to `40`

**Missing real motion:**
- Decrease `motion_threshold` (e.g. 5000 → 1000)
- Decrease blur kernel to `(11,11)`

**Clips are too short / cutting off the event:**
- Increase `record_duration`
- Add a cooldown: don't stop recording if motion is still active

**High CPU usage:**
- Process every Nth frame instead of every frame:
  ```python
  if frame_count % 2 == 0:  # process every other frame
      motion = self.detect_motion(prev_frame, current_frame)
  ```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Kernel dies immediately | `cv2.imshow()` called in headless env | Remove all `imshow`/`waitKey`/`destroyAllWindows` |
| `Could not open stream` | OpenCV missing FFmpeg build | Use `cv2.CAP_FFMPEG`; reinstall `opencv-python-headless` |
| `Could not extract stream URL` | yt-dlp outdated or video unavailable | Run `pip install -U yt-dlp`; check video is public |
| All frames trigger motion | Threshold too low or stream is live with constant change | Raise `motion_threshold`; increase blur |
| `.mp4` files unplayable | `mp4v` codec incompatibility | Try `avc1` fourcc or re-encode with ffmpeg post-run |

  
## 1. Overview  
This skill automates the process of **"plane spotting"** from live video feeds. It bridges the gap between raw video data and structured aviation intelligence by combining real-time frame extraction, computer vision object detection, and advanced Multimodal Large Language Models (MLLMs).  
## 2. Objectives  
* **Continuous Monitoring**: Capture snapshots from YouTube Live streams at set intervals without triggering bot-detection blocks.  
* **Precision Detection**: Isolate aircraft within complex airport backgrounds using YOLOv8.  
* **Expert Analysis**: Identify aircraft make, model, airline livery, and registration numbers using the Qwen2.5-VL vision-language model.  
## 3. System Architecture  
The pipeline operates in three distinct stages:  

| Stage | Tool | Description |
| --------- | --------------- | ------------------------------------------------------------------------------------------------------ |
| Ingestion | yt-dlp + ffmpeg | Streams live video through a pipe to bypass cookies/login requirements and saves frames. |
| Detection | YOLOv8n | Scans the frame for "Class 4" (aeroplane) and crops the image to the bounding box. |
| Reasoning | Qwen2.5-VL-7B | Analyzes the cropped image to extract tail numbers and airline specifics using an expert-tuned prompt. |
  
****4. Key Features****  
* **Anti-Bot Snapshotting**: Uses a subprocess pipe to avoid traditional browser-scraping limitations and LOGIN_REQUIRED errors.  
* **Resource Optimized**: Leverages YOLO for detection to reduce the token/image count sent to the more "expensive" Qwen model, ensuring it only analyzes relevant data.  
* **Visual Reporting**: Generates a consolidated report in the notebook featuring the cropped image alongside its AI-generated identification.  
## 5. Requirements  
* **Hardware**: NVIDIA GPU with 16GB+ VRAM (e.g., Tesla T4 or P100 on Kaggle).  
* **Libraries**: ultralytics, transformers, qwen-vl-utils, yt-dlp, ffmpeg.  
## 6. Usage  
1. **Configure URL**: Provide a valid YouTube Livestream URL for an airport.  
2. **Run Pipeline**: Execute the capture function to grab snapshots at your desired interval.  
3. **Review Summary**: View the **Visual Detection Summary** for immediate visual and textual identification.  
  

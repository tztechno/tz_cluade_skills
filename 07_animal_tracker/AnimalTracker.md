  
# AI Animal Tracker (skill.md)  
## 1. Overview  
This skill automatically detects specific animals from live wildlife streams (e.g., YouTube) to analyze and explain their species and behaviors in real-time. By combining high-speed object detection via **YOLOv8** with advanced visual reasoning via **Qwen2.5-VL**, it extracts structured data on subtle ecological changes that viewers might otherwise miss.  
## 2. Objectives  
* **24/7 Monitoring:** Periodically capture snapshots from live streams to monitor the activities of wildlife or pets.  
* **Multi-Species Detection:** Utilize YOLOv8 to accurately isolate individual animals from complex backgrounds, such as thickets or night-vision footage.  
* **Behavioral Analysis:** Use Qwen2.5-VL to provide detailed commentary on species names, physical traits (color, patterns), and current behaviors (feeding, sleeping, threatening, etc.).  
## 3. System Architecture  
The pipeline consists of the following three stages:  

| Stage | Tool | Description |
| --------- | --------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Ingestion | yt-dlp + ffmpeg | Pipes streams with low latency. Extracts high-quality frames at specified intervals while bypassing authentication hurdles. |
| Detection | YOLOv8n / v8x | Detects animal classes (bird, cat, dog, horse, bear, etc.) and crops the target at an optimal size. |
| Reasoning | Qwen2.5-VL-7B | Verbalizes species, inferred health status, and unique behaviors from cropped images. Generates context-aware commentary. |
  
****4. Key Features****  
* **Dynamic Cropping:** Even if an animal is small or at the edge of the frame, the system passes an optimized crop based on the YOLO bounding box to Qwen, significantly improving analysis accuracy.  
* **Behavior Labeling:** Goes beyond simple identification to determine specific behavioral contexts such as "grooming" or "hunting."  
* **Wildlife Optimized Prompting:** Assigns the role of a "Biologist" to Qwen to generate professional reports including scientific names and conservation status.  
## 5. Requirements  
* **Hardware:** NVIDIA GPU (16GB+ VRAM recommended for Qwen2.5-VL inference).  
* **Libraries:** ultralytics (YOLO), transformers, qwen-vl-utils, yt-dlp, ffmpeg-python.  
* **Model Weights:** YOLOv8n (speed-focused) or YOLOv8x (detection-focused for small objects).  
## 6. Usage  
1. **URL Configuration:** Input a YouTube Live URL for an African savanna, a rescue shelter, or a pet camera.  
2. **Detection Settings:** Set YOLO class filters according to the monitoring target (e.g., elephants, cats, birds).  
3. **Execution:** Run the pipeline and review the "Animal Observation Reports" generated at regular intervals, complete with analyzed images.  

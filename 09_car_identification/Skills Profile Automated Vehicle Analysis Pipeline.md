  
# Skills Profile: Automated Vehicle Analysis Pipeline  
This document details the technical stack and professional skills required to implement a Vision-Language Model (VLM) pipeline for vehicle detection and descriptive analysis.  
## 1. Core Technical Stack  
* **Programming Language:** Python 3.x  
* **Deep Learning Frameworks:**  
    * **PyTorch:** Managing tensors and model hardware acceleration (CUDA).  
    * **Ultralytics (YOLO11):** Real-time object detection and spatial localization.  
    * **Hugging Face Transformers:** Implementation of the **Qwen2.5-VL** (Vision-Language) model.  
* **Computer Vision Libraries:**  
    * **OpenCV (cv2):** Video stream handling, geometric transformations (rotation/affine), and image cropping.  
    * **PIL (Pillow):** Image format conversion for VLM compatibility.  
## 2. Key Engineering Competencies  
## A. Computer Vision & Preprocessing  
* **Video Manipulation:** Ability to navigate specific frame indices using CAP_PROP_POS_FRAMES.  
* **Spatial Transformation:** Applying rotation matrices to frames to correct camera orientation before inference.  
* **Region of Interest (ROI) Extraction:** Dynamically cropping detected objects based on bounding box coordinates for granular analysis.  
## B. Machine Learning Inference (VLM + Detection)  
* **Object Detection Logic:** Configuring confidence thresholds, IOU (Intersection over Union) thresholds, and class filtering (e.g., filtering for Cars and Trucks).  
* **Vision-Language Integration:** Utilizing **Qwen2.5-VL** to perform "Zero-shot" visual reasoning—extracting structured JSON data (brand, color, license plate) from raw pixels via natural language prompting.  
## C. Resource & Infrastructure Management  
* **CUDA Optimization:** Deploying models using device_map="auto" and torch_dtype="auto" to maximize GPU VRAM efficiency (critical for models like Qwen 7B).  
* **Environment Orchestration:** Managing dependencies within high-performance environments (e.g., Kaggle, Colab, or local GPU workstations).  
  
## 3. Workflow Logic Understanding  
An engineer working with this code must understand the **Three-Stage Pipeline**:  

| Stage   | Process    | Technology                                      |
| ------- | ---------- | ----------------------------------------------- |
| Stage 1 | Correction | Video frame extraction and rotation via OpenCV. |
| Stage 2 | Detection  | Localization of vehicles via YOLO11.            |
| Stage 3 | Cognition  | Detailed attribute analysis via Qwen2.5-VL.     |
  
****4. Hardware Requirements****  
* **GPU:** NVIDIA CUDA-enabled GPU (Minimum 16GB VRAM recommended for the 7B model).  
* **Drivers:** CUDA Toolkit and cuDNN compatible with PyTorch 2.x.  
  
## 5. Potential Skill Extensions (Future Tasks)  
* **Prompt Engineering:** Refining JSON output consistency from the VLM.  
* **Structured Data Parsing:** Implementing automated JSON validation for the VLM's string output.  
* **Batch Processing:** Optimizing the loop to handle multi-frame batches rather than single-frame inference.  

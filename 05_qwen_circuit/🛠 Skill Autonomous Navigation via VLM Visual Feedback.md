  
  
## 🛠 Skill: Autonomous Navigation via VLM Visual Feedback  
**Model:** Qwen2.5-VL-7B-Instruct  
**Environment:** MuJoCo Physics Engine (Circuit Task)  
## 📖 Description  
This skill enables a simulated vehicle to navigate a complex circuit by using a Vision Language Model (VLM) to interpret visual cues in real-time. Unlike traditional robotics which relies on pre-defined waypoints or geometric maps, this approach treats the VLM as a **zero-shot visual controller**.  
## 🏗 Architecture  
The system operates on a "Sense-Think-Act" loop:  
1. **Sense:** Captures a first-person "Chase Cam" image from MuJoCo, overlaid with distance sensors (raycasting).  
2. **Think:** Feeds the image to **Qwen2.5-VL** with a prompt to judge two specific variables:  
    * **Curve Direction:** Detecting the bend of the yellow dashed centerline.  
    * **Lateral Drift:** Determining if the car is centered or veering off-track.  
3. **Act:** Translates the VLM's JSON output (actions + confidence scores) into steering torque and velocity commands.  
## 🚀 Key Components  
* **Vision-Only Control:** No Feed-Forward (FF) or Feedback (FB) logic based on ground-truth coordinates.  
* **Confidence-Weighted Steering:** Steering magnitude is modulated by the VLM’s self-reported confidence ($curve\_conf$).  
* **Emergency Override:** A secondary raycasting safety layer provides hard-coded steering intervention if wall distance falls below $1.5m$.  
## 📊 Performance Metrics  
* **Target Velocity:** $18\text{ km/h}$ ($5\text{ m/s}$)  
* **Steering Limit:** $\pm 0.45\text{ radians}$  
* **Judgment Frequency:** Real-time inference per simulation interval.  
## 📝 Core Logic (JSON Protocol)  
The VLM is constrained to output a single JSON line to ensure low-latency parsing:  
JSON  
##   
##   
##   
##   
##   
##   
{  
**  "curve_direction": "CURVE_R",**  
**  "curve_action": "steer_right",**  
**  "curve_intensity": 0.8,**  
**  "curve_conf": 0.95,**  
**  "drift_action": "centered",**  
##   "drift_conf": 0.9  
## }  
## ⚠️ Constraints & Limitations  
* **Inference Latency:** Reliant on GPU compute (L4/A100) for real-time responsiveness.  
* **Visual Dependency:** Performance degrades if the "Yellow Centerline" is obscured or if lighting conditions change significantly.  
* **Stuck Detection:** Requires a 0.5s window of immobility to trigger a simulation reset.  
  
## How to use this skill  
1. **Environment Setup:** Install mujoco, transformers, and qwen-vl-utils.  
2. **Model Loading:** Initialize Qwen2.5-VL-7B-Instruct with device_map="auto".  
3. **Simulation:** Run the MuJoCo loop and pass the chase_cam renderer output to the query_qwen function.  
  

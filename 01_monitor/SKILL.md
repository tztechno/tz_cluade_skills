---
name: mac-process-monitor-generator
description: Generates a macOS-optimized real-time process monitor Python script using psutil and matplotlib. Outputs a runnable .py file that visualizes live CPU and memory usage of the top process.
metadata:
  version: 1.0.0
  target_os: macOS (darwin)
  dependencies: python>=3.9, psutil>=5.9.0, matplotlib>=3.7.0
---

# Mac Process Monitor Generator

This Skill generates a ready-to-run Python script that displays a real-time animated graph of the top CPU-consuming process on macOS.

## When to Use This Skill

Use this Skill whenever the user:
- Asks to monitor CPU or memory usage on Mac
- Wants to visualize process activity in real time
- Asks to "watch" which app is consuming the most resources
- Requests a system monitor, process watcher, or performance graph in Python
- Uses keywords like: `psutil`, `FuncAnimation`, `process monitor`, `top process`, `CPU graph`

## ⚠️ CRITICAL BEHAVIOR REQUIREMENT ⚠️

**DO NOT ask the user for preferences before generating.**
**DO NOT offer a menu of options.**
**DO NOT say "Would you like me to generate this?"**

**IMMEDIATELY:**
1. Generate the complete `.py` file
2. Save it to `/mnt/user-data/outputs/process_monitor.py`
3. Show how to install dependencies and run it
4. NO questions before delivering the file

---

## How It Works

### What the Generated Script Does

The script launches a live matplotlib window that:
- Tracks the **single top CPU-consuming process** updated every second
- Plots two subplots: **CPU (%)** and **Memory (%)** as scrolling time-series
- Displays process name and PID in the chart title
- Retains the last **60 seconds** of history

### macOS-Specific Constraints (MUST follow)

These are non-negotiable for the script to work on macOS:

| Issue | Required Fix |
|---|---|
| `cpu_percent()` returns 0 on first call | Call `psutil.cpu_percent(interval=None)` once before the animation loop |
| `cpu_percent` can return `None` | Always guard with `if cpu is not None` before comparison |
| matplotlib animation flickers | Set `cache_frame_data=False` in `FuncAnimation` |
| Process disappears mid-run | Wrap all `proc.info` access in `try/except (NoSuchProcess, AccessDenied, ZombieProcess)` |
| macOS event loop conflict | Use `plt.show()` (blocking) — do NOT use `plt.pause()` in a loop |

### Script Structure to Generate

```
1. Imports: psutil, matplotlib.pyplot, FuncAnimation, time
2. Config constants: UPDATE_INTERVAL=1000, HISTORY_LIMIT=60
3. Rolling data lists: times, cpu_values, mem_values
4. Initial cpu_percent warm-up call
5. get_current_top_cpu_process() function — safe iteration with None guard
6. Figure setup: 2 subplots (ax_cpu, ax_mem), sharex=True
7. update(frame) function — append data, trim to HISTORY_LIMIT, redraw both axes
8. FuncAnimation with cache_frame_data=False
9. plt.show()
```

### Customization Parameters (apply if user specifies)

| Parameter | Default | Description |
|---|---|---|
| `UPDATE_INTERVAL` | `1000` | Refresh rate in ms |
| `HISTORY_LIMIT` | `60` | Seconds of history shown |
| `figsize` | `(10, 8)` | Window size |
| CPU line color | `#FF3B30` | Apple red |
| Memory line color | `#007AFF` | Apple blue |

If the user requests different colors, intervals, or history windows — apply them. Otherwise use defaults.

---

## Behavior Guidelines

✅ **DO:**
- Always output a complete, runnable `.py` file — no placeholders
- Include all macOS guards listed above (None check, warm-up call, cache_frame_data)
- Save to `/mnt/user-data/outputs/process_monitor.py` and call `present_files`
- Show the install + run commands immediately after

✅ **Correct response pattern:**
```
"Here's the process monitor script for macOS."
[present_files with process_monitor.py]

Install dependencies:
  pip install psutil matplotlib

Run:
  python process_monitor.py
```

❌ **NEVER:**
- Ask "What update interval would you like?" before generating
- Generate a partial script with `# TODO` comments
- Omit the `cpu_percent is not None` guard
- Omit the warm-up `psutil.cpu_percent(interval=None)` call
- Use `plt.pause()` instead of `plt.show()`
- Set `cache_frame_data=True` or omit the parameter

---

## Example Prompts That Trigger This Skill

> "Mac でリアルタイムにCPUを監視するPythonスクリプトを作って"

> "psutil と matplotlib でトッププロセスのグラフを出したい"

> "process monitor script for macOS using FuncAnimation"

> "どのアプリがCPUを一番使ってるか可視化したい"

---

## Expected Output

**File:** `process_monitor.py`

**Window when run:**
- Title bar: `Target: Google Chrome (PID:1234)`
- Top chart: CPU % scrolling red line (0–100+ range, auto-scales)
- Bottom chart: Memory % scrolling blue line (0–20%+ range, auto-scales)
- X-axis: elapsed seconds since script launch
- Updates every 1 second, shows last 60 seconds

---

## Files

- `process_monitor.py` — Generated output script
- `requirements.txt` — `psutil>=5.9.0` and `matplotlib>=3.7.0`

## Notes

- Tested on macOS Ventura / Sonoma with Python 3.11
- Script requires no root/sudo privileges
- Some system processes may be hidden due to macOS SIP (System Integrity Protection) — this is expected behavior
- If `get_current_top_cpu_process()` returns `None` (e.g., immediately after boot), the chart shows 0 values safely

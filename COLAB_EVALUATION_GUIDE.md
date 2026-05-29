# Colab Evaluation Guide

## 🎯 Evaluation Workflow

### **Setup (5-10 minutes)**

```
1. Open colab_server.ipynb
   ├─ Run cells 1-5 (setup + start server)
   └─ Run cell 6 (start monitoring dashboard)

2. Open colab_edge_client_camera1.ipynb (in new tab)
   ├─ Run cells 1-4 (setup + config)
   └─ Run cell 5 (start Camera-1 edge client)

3. Open colab_edge_client_camera2.ipynb (in new tab)
   ├─ Run cells 1-4 (setup + config)
   └─ Run cell 5 (start Camera-2 edge client)

4. Open colab_edge_client_camera3.ipynb (in new tab)
   ├─ Run cells 1-4 (setup + config)
   └─ Run cell 5 (start Camera-3 edge client)
```

### **Monitoring (Real-time)**

While cameras are running:
- Server notebook: Watch **cell 6** dashboard for live stats
- Shows: Total assignments, gallery entries, recent matches
- Similarity scores tell you if matches are working

### **Evaluation (After cameras finish)**

```
5. Download crops from each camera
   ├─ Each client notebook → cell 7 (download zip)
   └─ Extract locally

6. Run offline evaluation (local or Colab)
   python evaluate.py --gt ground_truth.csv --category person
   └─ Produces: Rank-1, Rank-5, mAP, F1 curves
```

---

## 📊 What Each Notebook Does

| Notebook | Purpose | Input | Output |
|----------|---------|-------|--------|
| **colab_server.ipynb** | Central server | `/register_track` requests | Global IDs, logs |
| **colab_edge_client_camera1/2/3.ipynb** | Process video | Video file + detector + ReID model | Embeddings → server |
| (local) **evaluate.py** | Offline metrics | Ground truth CSV + crops | Rank-1, mAP, F1 |

---

## 🔄 Data Flow

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Camera-1 (Colab) │     │ Camera-2 (Colab) │     │ Camera-3 (Colab) │
│  edge_client.py  │     │  edge_client.py  │     │  edge_client.py  │
│                  │     │                  │     │                  │
│ 1. YOLO detect   │     │ 1. YOLO detect   │     │ 1. YOLO detect   │
│ 2. ByteTrack     │     │ 2. ByteTrack     │     │ 2. ByteTrack     │
│ 3. ReID extract  │     │ 3. ReID extract  │     │ 3. ReID extract  │
│                  │     │                  │     │                  │
│ embeddings ───────────→ server.py ←────── embeddings ←────── embeddings
│ (features)       │     │ (Flask)          │     │ (features)
│ (crops)          │     │ gallery_db + EMA │     │ (crops)
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                  │
                                  ├─→ global_id = "Person-1"
                                  ├─→ global_id = "Car-3"
                                  └─→ logs + monitor dashboard
```

---

## ✅ Configuration Checklist

### **Option A: All on Colab (Simplest)**
```
SERVER_URL in each client: http://localhost:5005/register_track
├─ All notebooks running on same Colab runtime
└─ ✓ Easy setup, ✓ Fast, ✓ Recommended for testing
```

### **Option B: Server Local + Clients on Colab (Advanced)**
```
SERVER_URL in each client: http://<your-local-ip>:5005/register_track
├─ Run server.py on your machine: python server.py
├─ Find your local IP: ipconfig (Windows) or ifconfig (Mac/Linux)
└─ ✓ More realistic, ✓ Better for production testing
```

### **Option C: Everything Local (Development)**
```
Run locally: python server.py
            python edge_client.py --camera-id Camera-1 --video-source test.mp4
            python edge_client.py --camera-id Camera-2 --video-source test.mp4
            python edge_client.py --camera-id Camera-3 --video-source test.mp4
└─ ✓ Fastest debugging, ✗ Can't use Colab GPU
```

---

## 📈 Expected Results

### **Server Monitoring (cell 6)**
```
╔══════════════════════════════════════════════════════════════════════╗
║ ReID Server Monitor (Updated: 12:34:56)
╠══════════════════════════════════════════════════════════════════════╣
║ Total Assignments: 145
║ Gallery Entries: {'0': 32, '2': 18}  (32 persons, 18 cars)
║ Track Counters: {'0': 32, '2': 18}
╠══════════════════════════════════════════════════════════════════════╣
║ Recent Assignments:
║   Camera-1     Person          Person-1        [matched ] (sim: 0.912)
║   Camera-2     Car             Car-3           [created ] (sim: N/A)
║   Camera-3     Person          Person-1        [matched ] (sim: 0.898)
║   Camera-1     Car             Car-3           [matched ] (sim: 0.876)
╚══════════════════════════════════════════════════════════════════════╝
```

### **Evaluation Metrics (evaluate.py)**
```
=== ReID Evaluation Results ===
Category: person
Dataset size: 250 person identities across 3 cameras

Ranking Metrics (Threshold-Free):
  Rank-1: 87.3%
  Rank-5: 96.2%
  mAP:    81.5%

Threshold Sweep (0.20–0.90):
  Best F1: 0.889 @ threshold=0.65
  Precision: 89.2%
  Recall: 88.7%
```

---

## 🐛 Troubleshooting

### **Server can't connect to clients**
- ✓ In client notebooks, change `SERVER_URL` to match your server
- ✓ Make sure server cell 5 is running first
- ✓ Check firewall if using Option B (local server)

### **Similarity scores too low (all < 0.5)**
- ✓ Check vehicle ReID weights are loading (should say "Successfully loaded pretrained weights")
- ✓ Verify videos have enough motion to extract distinct tracks
- ✓ Try adjusting `THRESHOLDS` in server (higher = stricter matching)

### **"Connection refused" in clients**
- ✓ Did you run server cells 5+6 first?
- ✓ Is the port 5005 available? (`lsof -i :5005` on Mac/Linux)
- ✓ Try restarting the server notebook

### **Out of memory (Colab)**
- ✓ Reduce video resolution or frame count
- ✓ Run only 1-2 cameras at a time
- ✓ Clear output (`Ctrl+/` in Colab) to free memory

---

## 📝 Evaluation Checklist

- [ ] Server running and accessible
- [ ] All 3 cameras connected and processing
- [ ] Monitor showing assignments and matches
- [ ] Crops saved from all 3 cameras
- [ ] Downloaded crops locally
- [ ] Created ground_truth.csv with cross-camera links
- [ ] Ran evaluate.py and got metrics
- [ ] Verified Rank-1 > 70% (good), > 85% (excellent)

---

## 🚀 Next Steps

1. **Quick test (5 min)**: Use Option A (all Colab), 1 short video, verify flow works
2. **Full evaluation**: Use 3 different videos (different lighting/angles), save all crops
3. **Optimize**: Adjust thresholds based on results, retrain if needed
4. **Production**: Move to Option B (local server) for real deployment


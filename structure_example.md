# Project structure
```
├── backend/        # FastAPI backend
|   ├── cfg/        # Config
|   |   ├── Trackers/       # Trackers' YAMLs
|   |   └── cfg.py      # Config file with the parameter for the processing pipeline
|   ├── mediamtx_logic/      # Solution for testing stream without rtsp camera. Usage guide is in face-recognition-demo repo
|   ├── sql/     # Not ready yet, DB must be reworked
|   ├── users/       # PseudoDB, contains image, cropped image and enbedding for each person
|   ├── utils/       # Utils
|   |   ├── models_logic/    # Models logic
|   |   |   ├── Models_layers_operations.py     # Custom layers for the converted models
|   |   |   ├── models_with_debug.py        # Models classes with included time tracking for searching bottlenecks
|   |   |   ├── models.py       # Should be used in prod. Not ready yet
|   |   |   └── SAHI.py     # SAHI algorithm logic
|   |   ├── alignment.py        # Face alignment
|   |   ├── custom_torch_components.py     # Deprecated Models_layers_operations
|   |   ├── drawing_utils.py        # Drawing utils
|   |   ├── gpu_monitor.py       # File that contains logic for measuring GPU load during the work
|   |   ├── image_utils.py      # For the face crop
|   |   └── models.py       # Should be used in prod instead of models_with_debug.py. Is not ready yet
|   |   └── other files are deprectated.
├── docs/       # Folder for docs, not used for now.
├── face-recognition-demo/      # SAYLESS repo for the frontend + old project implementation. Description of new files
|   ├── frontend/       # Frontend
|   |   ├── app/        # App
|   |   |   ├── live/       # Live page
|   |   ├── components/     # Components
|   |   |   ├── StreamVisualization.tsx     # Processing results visualization on the client side
├── go2rtc/     # Go2rtc logic. YAML with params
├── models/     # Models dir. Must be downloaded from google disc, Maxym Pechonkin must know. Also, inside must be an onnx/ dir.
├── tests/    # Tests. Not Unit tests
|   ├── check_rtsp_stream.py        # Check rtsp stream and reading time
|   ├── rtsp_bridge.py      # For testing stream with mediamtx.
|   ├── stream_api.py       # Main file for processing live stream. Check args in stream_processor.py
|   ├── stream_processor.py     # File with the logic for processing stream. Launch args are here
|   ├── STREAMING_README.md     # Process description, somehow deprecated
|   ├── test_pipeline_video_fps.py      # Main file for processing video. Spawns dir tests_results with all the info
|   ├── test_SAHI_detection.py      # Was used for testing SAHI logic. Deprecated
|   └── test_visual_detection_ONNX.py       # Description is in the first lines of the file
├── CLAUDE.md       # File for the CLAUDE CODE
└──
```
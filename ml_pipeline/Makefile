PYTHON ?= python3
CONFIG ?= config.yaml
COMPILE_MODE ?= auto

.PHONY: help install harvest autolabel prepare train compile compile-onnx compile-docker all test clean

help:
	@echo "Himawari-9 Band 13 Storm Segmentation ML Pipeline (Pi 5 + Hailo-8)"
	@echo ""
	@echo "Targets:"
	@echo "  make install         Install Python dependencies into current environment"
	@echo "  make harvest         Step 1: Download historical Himawari-9 Band 13 frames"
	@echo "  make autolabel       Step 2: Segment IR frames -> YOLOv8-seg labels + qa_preview.png"
	@echo "  make prepare         Step 3: Temporal 80/20 split, data.yaml, 256-frame calib_set.npy"
	@echo "  make train           Step 4: Train & validate yolov8s-seg + export static ONNX"
	@echo "  make compile         Step 5: Compile INT8 .hef (COMPILE_MODE=$(COMPILE_MODE))"
	@echo "  make compile-onnx    Step 5 (macOS/ARM64 helper): Prepare static ONNX + .alls only"
	@echo "  make compile-docker  Step 5 (x86_64 Linux): Run Hailo DFC inside Docker"
	@echo "  make all             Run full Steps 1-5 end-to-end via run_pipeline.py"
	@echo "  make test            Run automated pytest verification suite"
	@echo "  make clean           Remove generated work/ artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

harvest:
	$(PYTHON) 1_harvest_himawari.py --config $(CONFIG)

autolabel:
	$(PYTHON) 2_autolabel_seg.py --config $(CONFIG)

prepare:
	$(PYTHON) 3_prepare_dataset.py --config $(CONFIG)

train:
	$(PYTHON) 4_train_yolov8_seg.py --config $(CONFIG)

compile:
	$(PYTHON) 5_compile_hailo8.py --config $(CONFIG) --mode $(COMPILE_MODE)

compile-onnx:
	$(PYTHON) 5_compile_hailo8.py --config $(CONFIG) --mode onnx-only

compile-docker:
	./docker/compile_in_docker.sh

all:
	$(PYTHON) run_pipeline.py --config $(CONFIG) --compile-mode $(COMPILE_MODE)

test:
	$(PYTHON) -m pytest -v tests/

clean:
	rm -rf work/

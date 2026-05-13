@echo off

echo Creating conda environment...
conda create -n onnx_clean python=3.10 -y

echo Activating environment...
call conda activate onnx_clean

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing ONNX export dependencies...
pip install ^
ultralytics==8.4.42 ^
torch==2.1.2 ^
torchvision==0.16.2 ^
onnx==1.16.0 ^
onnxruntime==1.17.0 ^
onnxslim ^
numpy==1.26.4 ^
opencv-python==4.8.1.78

echo Exporting YOLO11 model to ONNX...
yolo export ^
model=best-stage2.pt ^
format=onnx ^
imgsz=640 ^
opset=17 ^
simplify=True ^
dynamic=False

echo Export completed!
pause
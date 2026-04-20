#!/bin/bash
# Script để tự động chạy pipeline trên server hoặc Kaggle VM

# Thoát ngay nếu có lỗi
set -e

echo "======================================"
echo "    BẮT ĐẦU AUTOCPD PIPELINE          "
echo "======================================"

# Cài đặt requirements nếu chưa có
echo "[1] Cài đặt dependencies..."
pip install -r requirements.txt || echo "pip install failed"

# Đảm bảo encoding
export PYTHONIOENCODING=utf-8

# Chạy pipeline theo thứ tự
echo "--------------------------------------"
echo "[2] Chạy Huấn Luyện (Training)"
python scripts/train.py

echo "--------------------------------------"
echo "[3] Chạy Đánh Giá (Evaluation)"
python scripts/evaluate.py

echo "--------------------------------------"
echo "[4] Chạy Phát Hiện (Detection)"
python scripts/detect.py

echo "======================================"
echo "   KẾT THÚC PIPELINE THÀNH CÔNG       "
echo "   Vui lòng kiểm tra thư mục outputs/ "
echo "======================================"

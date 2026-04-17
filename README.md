# Guideline

## References
[**Coreset Sampling from Open-Set for Fine-Grained Self-Supervised Learning**](https://arxiv.org/abs/2303.11101)  

---

## 1. Setup

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## 2. Environment Setup & Checkpoints

Trước khi chạy, bạn cần thiết lập đúng BASE_DIR và tải pretrained weights.

Tải pretrained checkpoint
```cd data
mkdir -p pretrained_ckpts
Example: 
wget --show-progress -O [BASE_DIR]/pretrained_ckpts/[DATA]_resnet50_pretrain_simclr_merge_imagenet_simcore.pth \
https://huggingface.co/sungnyun/openssl-simcore/resolve/main/aircraft_resnet50_pretrain_simclr_merge_imagenet_simcore.pth
```
Lưu ý:

Thay [BASE_DIR] bằng đường dẫn project của bạn
Thay [DATA] tương ứng với dataset:
aircraft,
cars,
pets,
cub,
flowers


## 3. Dataset Preparation
Theo đúng cấu hình của bài báo gốc, bạn cần tải các tập dữ liệu Fine-Grained và đặt chúng vào thư mục ./data/. Cách tải: Đọc file README trong folder ./data có hướng dẫn chi tiết

## 4. Training & Evaluation
Tạo file train.sh
```
#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

BASE_DIR="/path/to/your/project"
BATCH_SIZE=128
CKPT_DIR="${BASE_DIR}/pretrained_ckpts"

DATASETS=("aircraft" "cars" "pets" "cub" "flowers")

echo "Bắt đầu Training cho ${#DATASETS[@]} datasets..."
echo "====================================================="

for DATA in "${DATASETS[@]}"; do
    echo "====================================================="
    echo "ĐANG XỬ LÝ DATASET: [ ${DATA^^} ]"
    echo "====================================================="
    
    SIMCORE_CKPT="${CKPT_DIR}/${DATA}_resnet50_pretrain_simclr_merge_imagenet_simcore.pth"

    EVAL_TAG="eval_official_${DATA}"
    
    cd $BASE_DIR
    
    echo "Đang chạy Linear Evaluation cho ${DATA}..."
    python train_sup.py \
        --tag $EVAL_TAG \
        --dataset $DATA \
        --model resnet50 \
        --batch_size $BATCH_SIZE \
        --data_folder ./data/ \
        --pretrained \
        --pretrained_ckpt $SIMCORE_CKPT \
        --method simclr \
        --epochs 100 \
        --learning_rate 10 \
        --weight_decay 0 \
        --num_workers 16
        
    echo "XONG DATASET: [ ${DATA^^} ]!"
    echo "-----------------------------------------------------"
    
    sleep 5
done
```    
Chạy script
```chmod +x train.sh```
## 5. Evaluation

Khi training hoàn tất, kiểm tra:

Val Acc@1 (epoch cuối)
So sánh với paper gốc
Đánh giá chất lượng representation

import os
import torch
from collections import Counter
from torchvision import datasets
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

# ================= CẤU HÌNH =================
ckpt_path = "./save/aircraft_resnet50_pretrain_simclr_merge_imagenet_TEST_simcore_200_epochs/last.pth"
imagenet_dir = "./data/CLS-LOC/train"
# ============================================

print("1. Đang tải danh sách ảnh từ Checkpoint...")
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
indices = checkpoint.get('indices', [])

if not indices:
    print("Không tìm thấy selected_indices. Vui lòng kiểm tra lại file last.pth")
    exit()

print("2. Đang phân tích phân phối ID thư mục...")
dataset = datasets.ImageFolder(root=imagenet_dir)
idx_to_folder = {v: k for k, v in dataset.class_to_idx.items()}

# Ánh xạ từ Index của ảnh sang tên thư mục chứa ảnh (ví dụ: '48', '87')
selected_folders = [idx_to_folder[dataset.targets[idx]] for idx in indices]

# Đếm tần suất và lấy Top 20
counter = Counter(selected_folders)
top_20_folders = counter.most_common(20)

print("3. Đang triệu hồi AI 'Phiên dịch viên' (ResNet-50)...")
# Khởi tạo model và đẩy lên GPU cho nhanh
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights).to(device).eval()
preprocess = weights.transforms()

print("\n" + "="*70)
print(f"BẢNG XẾP HẠNG TOP 20 CLASS SAU KHI DỊCH TỰ ĐỘNG")
print("="*70)

translated_top_20 = []

# Tự động quét và dịch từng thư mục trong Top 20
for rank, (folder, count) in enumerate(top_20_folders, 1):
    folder_path = os.path.join(imagenet_dir, folder)
    images = os.listdir(folder_path)
    
    if not images:
        category_name = f"Unknown_Folder_{folder}"
    else:
        # Lấy ảnh đầu tiên trong thư mục để AI nhìn thử
        img_path = os.path.join(folder_path, images[0])
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert('RGB')
                batch = preprocess(img_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                prediction = model(batch).squeeze(0)
            
            class_id = prediction.argmax().item()
            # Lấy tên tiếng Anh thực sự của class đó
            category_name = weights.meta["categories"][class_id]
        except Exception as e:
            category_name = f"Error_reading_{folder}"

    # In ra terminal
    percentage = (count / len(indices)) * 100
    print(f"Top {rank:02d} | Thư mục {folder:<4} -> {category_name:<25} | {count:>5} ảnh ({percentage:.2f}%)")
    
    # Lưu lại để vẽ biểu đồ
    translated_top_20.append((category_name.capitalize(), count))

print("="*70)

# 4. Vẽ biểu đồ mới với tên class rõ ràng
print("\n4. Đang vẽ biểu đồ thống kê...")
labels = [item[0] for item in translated_top_20]
counts = [item[1] for item in translated_top_20]

plt.figure(figsize=(12, 8))
# Vẽ ngang và lật ngược mảng để Top 1 nằm trên cùng
plt.barh(labels[::-1], counts[::-1], color='coral') 
plt.xlabel('Số lượng ảnh được chọn')
plt.title(f'Top 20 Classes được SimCore chọn nhiều nhất (Tổng: {len(indices)} ảnh)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

output_file = "auto_decoded_distribution.png"
plt.savefig(output_file, dpi=300)
print(f"Hoàn tất! Đã lưu biểu đồ tự động dịch vào file: {output_file}")
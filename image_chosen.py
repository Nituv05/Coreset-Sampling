import os
import torch
import shutil
import random
from torchvision import datasets

ckpt_path = "/mnt/disk3/tinvt/openssl-simcore/save/aircraft_resnet50_pretrain_simclr_merge_imagenet_TEST_simcore_200_epochs/last.pth"

# 2. Đường dẫn gốc tới tập ImageNet
imagenet_dir = "./data/CLS-LOC/train"

# 3. Thư mục bạn muốn xuất ảnh ra để xem
output_dir = "./view_coreset_samples_200_epochs"
# os.makedirs(output_dir, exist_ok=True)

print("Đang tải checkpoint...")
# Nạp checkpoint (nhớ tắt weights_only để không bị lỗi bảo mật)
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Trích xuất danh sách index đã được SimCore chọn
indices = checkpoint.get('indices', [])

if not indices:
    print("Không tìm thấy selected_indices trong file checkpoint này!")
    exit()

print(f" Đã tìm thấy {len(indices)} ảnh được SimCore chọn.")
print("Đang đối chiếu thư mục ImageNet (sẽ mất khoảng vài chục giây)...")

# Khởi tạo lại ImageFolder để lấy mapping giữa index và file path
dataset = datasets.ImageFolder(root=imagenet_dir)

# Chọn ngẫu nhiên 100 ảnh để copy ra xem (bạn có thể tăng giảm số này)
num_view = min(1000, len(indices))
sample_indices = random.sample(list(indices), num_view)

print(f"Đang copy {num_view} ảnh ngẫu nhiên ra thư mục '{output_dir}'...")
for i, idx in enumerate(sample_indices):
    # dataset.samples[idx] trả về tuple (đường_dẫn_file, id_class)
    img_path = dataset.samples[idx][0]
    file_name = os.path.basename(img_path)
    class_folder = os.path.basename(os.path.dirname(img_path)) 
    
    # Đổi tên file lưu ra: thêm số thứ tự và tên class gốc của ImageNet
    new_name = f"{i:03d}_{class_folder}_{file_name}"
    dest_path = os.path.join(output_dir, new_name)
    
    shutil.copy2(img_path, dest_path)

print(f"Hoàn tất! Hãy mở thư mục '{output_dir}' để ngắm thành quả.")
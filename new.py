import os

# Thư mục sau khi giải nén
extract_path = "dataset"

# In ra toàn bộ cây thư mục và file để kiểm tra
for root, dirs, files in os.walk(extract_path):
    level = root.replace(extract_path, "").count(os.sep)
    indent = " " * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 4 * (level + 1)
    for f in files[:5]:  # chỉ in 5 file đầu để gọn
        print(f"{subindent}{f}")

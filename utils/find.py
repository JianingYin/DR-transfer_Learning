from PIL import Image
import os
# 跑一下这个简单的检查
for root, dirs, files in os.walk(r'F:\DR\Data\aptos\aptos_split'):
    for file in files:
        if file.endswith('.png'):
            img_path = os.path.join(root, file)
            try:
                with Image.open(img_path) as img:
                    img.load() 
            except Exception as e:
                print(f"损坏的文件: {img_path}, 错误: {e}")
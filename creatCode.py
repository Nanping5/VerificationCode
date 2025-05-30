import random
import os
from captcha.image import ImageCaptcha
from PIL import Image

# 定义所有可能的字符（数字0-9 + 大小写字母）
CHARACTERS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_COUNT = 4  # 每个验证码包含4个字符
NUM_CHARS = len(CHARACTERS)

def generate_captcha(output_dir, num_images):
    """生成指定数量的验证码图片到指定目录"""
    # 添加基础路径
    output_dir = os.path.join("./", output_dir)
    os.makedirs(output_dir, exist_ok=True)  # 创建对应目录
    captcha_generator = ImageCaptcha(width=160, height=60)
    
    generated_count = 0
    while generated_count < num_images:
        captcha_text = ''.join([random.choice(CHARACTERS) for _ in range(CHAR_COUNT)])
        file_path = os.path.join(output_dir, f'{captcha_text}.png')
        
        # 跳过已存在的文件（避免覆盖）
        if os.path.exists(file_path):
            continue
            
        image = captcha_generator.generate_image(captcha_text)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(file_path)
        generated_count += 1

if __name__ == "__main__":
    # 生成不同数据集
    generate_captcha(output_dir='train', num_images=50000)  # 训练集
    generate_captcha(output_dir='test', num_images=10000)   # 测试集
    generate_captcha(output_dir='val', num_images=10000)     # 验证集
    
    print("数据集生成完成：")
    print(f"- 训练集：{len(os.listdir('train'))} 张")
    print(f"- 测试集：{len(os.listdir('test'))} 张") 
    print(f"- 验证集：{len(os.listdir('val'))} 张")

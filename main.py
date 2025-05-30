import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, GlobalAvgPool2D, Dropout, Resizing
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard
from tensorflow.keras.utils import Sequence
from PIL import Image
import numpy as np
import string
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import traceback
# 在initialize_system()开头添加
os.environ.pop('TF_XLA_FLAGS',  None)  # 移除旧变量
# 基础配置
mpl.use('Agg')  # 非交互式后端
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志输出

# 模型参数
CAPTCHA_LENGTH = 4
BATCH_SIZE =128
EPOCHS = 80
WIDTH, HEIGHT = 160, 60
characters = string.digits + string.ascii_letters
num_classes = len(characters)

# 路径配置
DATA_DIR = Path("E:/VerificationCode")
MODEL_DIR = Path("E:/VerificationCode/Models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# 自定义回调 - 完全避开JSON序列化
class SafeModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', save_best_only=True):
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor, float('inf'))
        if not self.save_best_only or current < self.best:
            self.best = current
            # 使用HDF5格式直接保存权重
            self.model.save_weights(str(self.filepath))
            print(f"\n模型权重已保存到 {self.filepath}")


class SafeHistoryLogger(Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = Path(filepath)
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(float(v))

        # 使用pickle二进制格式保存
        with open(str(self.filepath), 'wb') as f:
            pickle.dump(self.history, f)


# 数据集加载器
class CaptchaSequence(Sequence):
    def __init__(self, directory, batch_size, steps, augment=False):
        self.directory = Path(directory)
        self.batch_size = batch_size
        self.steps = steps
        self.augment = augment
        self.image_paths = [f for f in self.directory.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]

        if not self.image_paths:
            raise ValueError(f"目录中没有图像文件: {directory}")

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_paths = random.sample(self.image_paths, self.batch_size)
        images, labels = [], []

        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')

                # 数据增强
                if self.augment:
                    angle = random.uniform(-5, 5)
                    img = img.rotate(angle, fillcolor=(255, 255, 255))

                    scale = random.uniform(0.95, 1.05)
                    new_w, new_h = int(WIDTH * scale), int(HEIGHT * scale)
                    img = img.resize((new_w, new_h))

                    x = random.randint(0, max(0, new_w - WIDTH))
                    y = random.randint(0, max(0, new_h - HEIGHT))
                    img = img.crop((x, y, x + WIDTH, y + HEIGHT))
                else:
                    img = img.resize((WIDTH, HEIGHT))

                images.append(np.array(img) / 255.0)

                # 标签处理
                captcha_text = img_path.stem[:CAPTCHA_LENGTH]  # 确保长度正确
                label = np.zeros((CAPTCHA_LENGTH, num_classes), dtype=np.float32)
                for j, ch in enumerate(captcha_text):
                    pos = characters.find(ch)
                    if pos != -1:
                        label[j, pos] = 1
                labels.append(label)

            except Exception as e:
                print(f"加载图片失败: {img_path}, 错误: {str(e)}")
                continue

        # 确保返回numpy数组
        return np.array(images), {f'out{i}': np.array(labels)[:, i, :] for i in range(CAPTCHA_LENGTH)}


# 模型构建
def create_model():
    inputs = Input((HEIGHT, WIDTH, 3))
    x = Resizing(HEIGHT, WIDTH)(inputs)

    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_tensor=x
    )

    # 冻结前100层
    for layer in base_model.layers[:100]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAvgPool2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.25)(x)

    outputs = [
        Dense(num_classes, activation='softmax', name=f'out{i}')(x)
        for i in range(CAPTCHA_LENGTH)
    ]

    return Model(inputs, outputs)


# 训练函数
def train_model():
    # 数据准备
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"

    train_size = len(list(train_dir.glob('*')))
    val_size = len(list(val_dir.glob('*')))

    train_steps = max(1, train_size // BATCH_SIZE)
    val_steps = max(1, val_size // BATCH_SIZE)

    print(f"训练样本数: {train_size}, 训练步数: {train_steps}")
    print(f"验证样本数: {val_size}, 验证步数: {val_steps}")

    train_seq = CaptchaSequence(train_dir, BATCH_SIZE, train_steps, augment=True)
    val_seq = CaptchaSequence(val_dir, BATCH_SIZE, val_steps)

    # 模型构建
    model = create_model()

    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 回调配置
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        SafeModelCheckpoint(MODEL_DIR / 'model_weights.h5'),
        SafeHistoryLogger(MODEL_DIR / 'training_history.pkl'),
        TensorBoard(log_dir=str(MODEL_DIR / 'logs'))
    ]

    # 训练
    history = model.fit(
        train_seq,
        epochs=EPOCHS,
        validation_data=val_seq,
        callbacks=callbacks,
        verbose=1
    )

    # 训练完成后保存完整模型（使用SavedModel格式）
    tf.saved_model.save(model, str(MODEL_DIR / 'saved_model'))
    return model, history


# 预测函数
def predict_captcha(model, image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((WIDTH, HEIGHT))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    return ''.join([characters[np.argmax(pred[0])] for pred in predictions])


# 主函数
def main():
    model_path = MODEL_DIR / 'saved_model'

    try:
        if model_path.exists():
            print("加载预训练模型...")
            model = tf.keras.models.load_model(str(model_path))
        else:
            print("训练新模型...")
            model, _ = train_model()
            print("模型训练完成")

        # 测试模型
        test_dir = DATA_DIR / "test"
        for img_path in list(test_dir.glob('*'))[:5]:
            try:
                prediction = predict_captcha(model, str(img_path))
                true_label = img_path.stem[:CAPTCHA_LENGTH]

                print(f"\n图片: {img_path.name}")
                print(f"真实标签: {true_label}")
                print(f"预测结果: {prediction}")
                print(f"结果: {'正确' if prediction == true_label else '错误'}")

                # 保存预测结果图
                plt.figure()
                plt.imshow(Image.open(img_path))
                plt.title(f" 真实: {true_label}\n预测: {prediction}")
                plt.axis('off')
                plt.savefig(MODEL_DIR / f"pred_{img_path.stem}.png")
                plt.close()

            except Exception as e:
                print(f"处理图片 {img_path} 时出错: {str(e)}")
                traceback.print_exc()

    except Exception as e:
        print(f"程序出错: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    # 设置GPU内存增长
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU配置错误: {str(e)}")

    main()
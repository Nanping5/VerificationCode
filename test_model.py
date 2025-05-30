import os
import tensorflow as tf
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import logging
from typing import Tuple, Dict
import string
os.environ.pop('TF_XLA_FLAGS',  None)  # 移除旧变量
# 配置继承
DATA_DIR = Path("E:/VerificationCode")
MODEL_DIR = Path("E:/VerificationCode/Models")
RESULTS_DIR = MODEL_DIR / "results"
WIDTH, HEIGHT = 160, 60
CAPTCHA_LENGTH = 4
characters = string.digits + string.ascii_letters

# 创建结果目录
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 配置日志
logging.basicConfig(filename=str(RESULTS_DIR / 'test_log.log'),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class SavedModelTester:
    def __init__(self, model_dir: Path):
        """初始化SavedModel测试器"""
        self.model_dir = model_dir
        self.model = None
        self.predict_fn = None
        self.load_model()

    def load_model(self):
        """加载SavedModel"""
        try:
            self.model = tf.saved_model.load(str(self.model_dir))
            if 'serving_default' in self.model.signatures:
                self.predict_fn = self.model.signatures['serving_default']
            else:
                self.predict_fn = list(self.model.signatures.values())[0]
            logging.info(" 模型加载成功")
        except Exception as e:
            logging.error(f" 加载模型失败: {str(e)}")
            raise

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """预处理图像"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((WIDTH, HEIGHT))
            img_array = np.array(img) / 255.0
            return img_array.astype(np.float32)
        except Exception as e:
            logging.error(f" 预处理失败 {image_path}: {str(e)}")
            return None

    def predict(self, image_path: Path) -> Tuple[str, Dict[int, bool]]:
        """执行预测并返回结果"""
        img_array = self.preprocess_image(image_path)
        if img_array is None:
            return None, None

        true_label = image_path.stem[:CAPTCHA_LENGTH]
        input_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0))

        try:
            outputs = self.predict_fn(input_tensor)
            prediction = ''.join([characters[np.argmax(outputs[f'out{i}'][0])]
                                  for i in range(CAPTCHA_LENGTH)])

            # 计算每个字符是否正确
            char_results = {
                i: (prediction[i] == true_label[i])
                for i in range(CAPTCHA_LENGTH)
            }

            return prediction, char_results
        except Exception as e:
            logging.error(f" 预测失败 {image_path}: {str(e)}")
            return None, None


def evaluate_dataset(tester: SavedModelTester, dataset_name: str) -> pd.DataFrame:
    """评估指定数据集并保存结果"""
    dataset_dir = DATA_DIR / dataset_name
    image_paths = list(dataset_dir.glob('*.[pj][np]g'))

    # 收集结果
    results = []
    for img_path in image_paths:
        true_label = img_path.stem[:CAPTCHA_LENGTH]
        prediction, char_results = tester.predict(img_path)

        if prediction is not None:
            results.append({
                'filename': img_path.name,
                'true_label': true_label,
                'prediction': prediction,
                'overall_correct': (prediction == true_label),
                **{f'char_{i}_correct': char_results[i]
                   for i in range(CAPTCHA_LENGTH)}
            })

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 保存原始结果
    result_file = RESULTS_DIR / f'{dataset_name}_results.csv'
    df.to_csv(result_file, index=False)
    logging.info(f"{dataset_name} 结果已保存到 {result_file}")

    return df


def generate_accuracy_tables(val_df: pd.DataFrame, test_df: pd.DataFrame):
    """生成准确率表格"""
    # 验证集准确率
    val_acc = {
        'overall': val_df['overall_correct'].mean()
    }
    for i in range(CAPTCHA_LENGTH):
        val_acc[f'char_{i}'] = val_df[f'char_{i}_correct'].mean()

    # 测试集准确率
    test_acc = {
        'overall': test_df['overall_correct'].mean()
    }
    for i in range(CAPTCHA_LENGTH):
        test_acc[f'char_{i}'] = test_df[f'char_{i}_correct'].mean()

    # 创建表格
    accuracy_df = pd.DataFrame({
        'Position': ['Overall'] + [f'Char_{i}' for i in range(CAPTCHA_LENGTH)],
        'Validation': [val_acc['overall']] + [val_acc[f'char_{i}'] for i in range(CAPTCHA_LENGTH)],
        'Test': [test_acc['overall']] + [test_acc[f'char_{i}'] for i in range(CAPTCHA_LENGTH)]
    })

    # 保存表格
    table_file = RESULTS_DIR / 'accuracy_tables.xlsx'
    with pd.ExcelWriter(table_file) as writer:
        accuracy_df.to_excel(writer, sheet_name='Accuracy Summary', index=False)

        # 详细结果表
        val_details = val_df[['filename', 'true_label', 'prediction', 'overall_correct'] +
                             [f'char_{i}_correct' for i in range(CAPTCHA_LENGTH)]]
        val_details.to_excel(writer, sheet_name='Validation Details', index=False)

        test_details = test_df[['filename', 'true_label', 'prediction', 'overall_correct'] +
                               [f'char_{i}_correct' for i in range(CAPTCHA_LENGTH)]]
        test_details.to_excel(writer, sheet_name='Test Details', index=False)

    logging.info(f" 准确率表格已保存到 {table_file}")
    return accuracy_df


def plot_accuracy_charts(val_df: pd.DataFrame, test_df: pd.DataFrame):
    """绘制准确率图表"""
    plt.figure(figsize=(18, 8))

    # 验证集准确率
    plt.subplot(1, 2, 1)
    val_acc = [val_df[f'char_{i}_correct'].mean() for i in range(CAPTCHA_LENGTH)]
    positions = range(CAPTCHA_LENGTH)

    plt.bar(positions, val_acc, color='skyblue')
    plt.xticks(positions, [f'Position {i}' for i in positions])
    plt.ylim(0, 1)
    plt.title('Validation  Set - Per Character Accuracy')
    plt.xlabel('Character  Position')
    plt.ylabel('Accuracy')

    # 添加总体准确率
    overall_val = val_df['overall_correct'].mean()
    plt.axhline(y=overall_val, color='red', linestyle='--',
                label=f'Overall: {overall_val:.2%}')
    plt.legend()

    # 测试集准确率
    plt.subplot(1, 2, 2)
    test_acc = [test_df[f'char_{i}_correct'].mean() for i in range(CAPTCHA_LENGTH)]

    plt.bar(positions, test_acc, color='lightgreen')
    plt.xticks(positions, [f'Position {i}' for i in positions])
    plt.ylim(0, 1)
    plt.title('Test  Set - Per Character Accuracy')
    plt.xlabel('Character  Position')
    plt.ylabel('Accuracy')

    # 添加总体准确率
    overall_test = test_df['overall_correct'].mean()
    plt.axhline(y=overall_test, color='red', linestyle='--',
                label=f'Overall: {overall_test:.2%}')
    plt.legend()

    # 保存图表
    chart_file = RESULTS_DIR / 'accuracy_charts.png'
    plt.tight_layout()
    plt.savefig(chart_file)
    plt.close()
    logging.info(f" 准确率图表已保存到 {chart_file}")


def generate_confusion_matrices(df: pd.DataFrame, dataset_name: str):
    """生成混淆矩阵"""
    plt.figure(figsize=(15, 5 * CAPTCHA_LENGTH))

    for i in range(CAPTCHA_LENGTH):
        plt.subplot(CAPTCHA_LENGTH, 1, i + 1)
        true_chars = df['true_label'].str[i]
        pred_chars = df['prediction'].str[i]

        cm = confusion_matrix(true_chars, pred_chars, labels=list(characters))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(characters), yticklabels=list(characters))
        plt.title(f'{dataset_name}  Set - Position {i} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # 保存混淆矩阵
    cm_file = RESULTS_DIR / f'{dataset_name}_confusion_matrices.png'
    plt.tight_layout()
    plt.savefig(cm_file)
    plt.close()
    logging.info(f"{dataset_name} 混淆矩阵已保存到 {cm_file}")


def main():
    """主测试流程"""
    try:
        # 初始化GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # 加载模型
        tester = SavedModelTester(MODEL_DIR / 'saved_model')

        # 评估验证集和测试集
        val_df = evaluate_dataset(tester, 'val')
        test_df = evaluate_dataset(tester, 'test')

        # 生成准确率表格
        accuracy_df = generate_accuracy_tables(val_df, test_df)

        # 绘制准确率图表
        plot_accuracy_charts(val_df, test_df)

        # 生成混淆矩阵
        generate_confusion_matrices(val_df, 'validation')
        generate_confusion_matrices(test_df, 'test')

        # 打印摘要
        print("\n测试结果摘要:")
        print(accuracy_df.to_string(index=False))

        print(f"\n详细结果已保存到: {RESULTS_DIR}")

    except Exception as e:
        logging.error(f" 测试失败: {str(e)}")
        print(f"测试过程中出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()
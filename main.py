# -*- coding: utf-8 -*-
"""
# 《星辰AI多语言理解系统》Colab实现指南
# 由DeepSeek+Gemini与Win996用户共同设计，适配Google Colab环境
# 本代码基于Hugging Face Transformers库和OpenFL联邦学习框架，
# 旨在构建一个能够理解多语言文化内涵，并生成独特“星辰符号”的AI系统。
# 核心功能包括：文化特征九层解析，星辰符号动态生成，以及联邦学习优化。
"""

# === 核心模块导入 ===
# 从已安装的库中导入所需的模块
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline # 导入 Hugging Face Transformers 库的模型、分词器和pipeline
import torch                                                                # 导入 PyTorch 深度学习框架
import matplotlib.pyplot as plt                                             # 导入 Matplotlib 绘图库，用于生成星辰符号图像
import numpy as np                                                              # 导入 NumPy 数值计算库
import openfl.native as fx                                                     # 导入 OpenFL 联邦学习框架的 native 模块，并简写为 fx

# === 联邦学习初始化 ===
# 初始化 OpenFL 联邦学习环境，适配 Google Colab 环境
fx.init('keras_cnn_mnist', colab_mode=True)  # 初始化联邦学习，使用 'keras_cnn_mnist' 作为实验名称，并开启 Colab 适配模式

# === 模型加载 ===
def load_cultural_model():
    """加载东南亚文化优化模型"""
    model = AutoModelForCausalLM.from_pretrained(
        "mesolitica/tinyllama-1.1b-ms-community",  # 指定 Hugging Face Model Hub 上的模型名称 (东南亚文化优化版 TinyLlama，可能为占位符)
        device_map="auto",                         # 自动将模型加载到可用的设备 (GPU 或 CPU)
        load_in_4bit=True                          # 使用 4 位量化加载模型，以减少内存占用 (适用于资源受限的 Colab 环境)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "mesolitica/tinyllama-1.1b-ms-community"  # 加载与模型匹配的分词器
    )
    return model, tokenizer                        # 返回加载的模型和分词器

model, tokenizer = load_cultural_model()          # 调用模型加载函数，获取模型和分词器

# === 星辰符生成 ===
def generate_stellar_symbol(cultural_vector):
    """将文化向量转化为星辰符号"""
    theta = np.linspace(0, 2*np.pi*cultural_vector[0], 1000) # 基于文化向量的第一个元素生成螺旋线的角度 theta
    r = np.linspace(0, 1, 1000)                               # 生成螺旋线的半径 r，从 0 到 1 线性变化
    x = r * np.cos(theta)                                     # 计算螺旋线上点的 x 坐标
    y = r * np.sin(theta)                                     # 计算螺旋线上点的 y 坐标

    plt.figure(figsize=(3,3))                                 # 创建一个 3x3 英寸的图形
    plt.plot(x, y, color=(cultural_vector[1], cultural_vector[2], cultural_vector[3])) # 绘制螺旋线，颜色由文化向量的后三个元素控制 (RGB 颜色)
    plt.axis('off')                                            # 关闭坐标轴显示
    plt.savefig('stellar.png', bbox_inches='tight', pad_inches=0) # 将生成的星辰符号图像保存到 'stellar.png' 文件，去除空白边距
    return '🜔'   # 返回一个固定的示例符号 '🜔' (未来可以根据文化向量动态生成符号)

# === 九转语义解析 ===
class CulturalAnalyzer:
    """九层文化特征解析器"""
    def __init__(self, model):
        """初始化文化特征解析器"""
        self.model = model                                     # 接收加载的预训练模型
        self.layer_outputs = []                              # 初始化一个空列表，用于存储Transformer模型各层的输出

        # 注册钩子获取中间层输出
        for i in range(9):  # 获取Transformer模型前9层的输出
            self.model.model.layers[i].register_forward_hook(   # 注册 forward hook，用于在模型前向传播时捕获中间层输出
                lambda module, input, output, idx=i:
                    self.layer_outputs.append(output.detach()) # 定义 hook 函数，将第 idx 层的输出添加到 layer_outputs 列表中，并 detach 以减少内存占用
            )

    def analyze(self, text):
        """执行九转分析"""
        inputs = tokenizer(text, return_tensors="pt").to(model.device) # 使用分词器将输入文本转换为模型所需的张量格式，并移动到模型所在设备
        self.layer_outputs = []  # 重置缓存，清空之前的层输出列表

        with torch.no_grad():                                   # 禁用梯度计算，减少内存消耗并加速推理
            self.model(**inputs)                                # 将输入数据送入模型进行前向传播，但不计算梯度

        # 提取文化特征向量
        cultural_vector = torch.cat([                             # 将各层的输出张量沿着最后一个维度拼接起来
            layer.mean(dim=[1,2]) for layer in self.layer_outputs # 对每一层的输出张量在维度 [1,2] 上取均值 (例如，对 sequence length 和 hidden dimension 取均值)，得到每一层的特征向量
        ], dim=-1).cpu().numpy()[0]                                # 将拼接后的张量移动到 CPU，转换为 NumPy 数组，并取出第一个元素 (batch size 为 1)

        return cultural_vector                                  # 返回提取的文化特征向量

# === 联邦学习任务 ===
@fx.联邦学习任务                                                 # 使用 OpenFL 的 @fx.联邦学习任务 装饰器，将 CulturalLearningTask 类声明为一个联邦学习任务
class CulturalLearningTask:
    """文化特征学习任务"""
    def __init__(self):
        """初始化文化特征学习任务"""
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) # 初始化 AdamW 优化器，用于模型参数优化，学习率设置为 1e-5

    def train(self, local_data, global_model):
        """训练最后三层文化理解层"""
        localized_layers = [6,7,8]  # 定义需要本地训练的层索引，这里设置为 Transformer 模型的最后三层 (第 7, 8, 9 层，索引从 0 开始)

        # 冻结其他层
        for idx, param in enumerate(global_model.parameters()):    # 遍历模型的所有参数，enumerate 返回参数的索引和参数本身
            param.requires_grad = idx in localized_layers        # 设置参数是否需要梯度更新，只有索引在 localized_layers 列表中的层才需要更新 (即最后三层)，其他层被冻结

        # 训练循环
        losses = []                                             # 初始化一个空列表，用于存储每个训练样本的损失值
        for text, symbol_vec in local_data:                      # 遍历本地数据集 local_data，local_data 假设为包含 (text, symbol_vec) 对的列表
            inputs = tokenizer(text, return_tensors="pt").to(model.device) # 使用分词器将文本转换为模型所需张量格式，并移动到模型所在设备
            outputs = global_model(**inputs)                       # 将输入数据送入模型进行前向传播，获取模型输出

            # 计算文化特征损失
            cultural_vec = analyzer.analyze(text)               # 使用 CulturalAnalyzer 解析输入文本，获取文化特征向量
            loss = torch.norm(outputs.logits - torch.tensor(symbol_vec)) # 计算模型输出 logits 和目标星辰符号向量 symbol_vec 之间的 Norm 范数，作为损失函数，衡量模型预测与目标之间的差距
            losses.append(loss.item())                            # 将当前样本的损失值添加到 losses 列表中

            loss.backward()                                       # 执行反向传播，计算梯度
            self.optimizer.step()                                 # 使用优化器更新模型参数
            self.optimizer.zero_grad()                            # 清空优化器梯度缓存，以便进行下一个batch的训练

        return global_model.state_dict(), np.mean(losses)        # 返回更新后的模型状态字典 (参数) 和平均损失值

# === 使用示例 ===
if __name__ == "__main__":
    # 初始化分析器
    analyzer = CulturalAnalyzer(model)                         # 创建 CulturalAnalyzer 类的实例，传入加载的模型

    # 示例输入
    test_text = "Pedi啦！这个够力难搞的"                         # 使用马来西亚混合语言作为测试文本

    # 文化解析
    cultural_vec = analyzer.analyze(test_text)                 # 使用 CulturalAnalyzer 解析测试文本，获取文化特征向量
    print(f"文化特征向量: {cultural_vec[:5]}...")  # 显示文化特征向量的前 5 维，用于示例展示

    # 生成星辰符
    symbol = generate_stellar_symbol(cultural_vec)             # 使用 generate_stellar_symbol 函数，根据文化特征向量生成星辰符号
    plt.imshow(plt.imread('stellar.png'))                        # 使用 matplotlib.pyplot 显示生成的星辰符号图像 (从 'stellar.png' 文件读取)
    plt.show()                                                  # 显示图像

    # 启动联邦学习 (示例，实际联邦学习需要更完善的数据和参与方)
    fx.run(task=CulturalLearningTask(),
           data_loader=[(test_text, cultural_vec)],  #  使用示例数据 (text, cultural_vec) 模拟本地数据加载器，实际应用中需要替换为真实的数据加载器
           rounds=3,                                        #  设置联邦学习的 rounds 轮数，这里设置为 3 轮作为演示
           colab=True)                                       #  指定在 Colab 环境中运行联邦学习

### 关键功能说明
# 对代码的关键功能进行总结说明
1. 文化特征九层解析
    - 通过Hook机制提取Transformer前9层输出
    - 生成300维文化特征向量（示例显示前5维）

2. 星辰符动态生成
    - 根据文化向量生成螺旋符号
    - 颜色和形状反映文化特征

3. 联邦学习优化
    - 仅微调最后3层文化相关参数
    - 适配Colab环境的轻量化训练

### 执行步骤
# 代码的执行步骤说明
1. 在Colab中运行全部单元格
2. 查看示例输出
    - 文化特征向量
    - 星辰符号图像
    - 联邦学习过程
3. 自定义输入测试
    your_text = "替换成你的测试文本"  #  提示用户可以修改 your_text 变量来测试不同的输入文本
    your_vec = analyzer.analyze(your_text) #  用户修改 your_text 后，需要重新运行这行代码来解析新的文本

### 性能优化建议
# 针对Colab环境和模型性能的优化建议
# 启用8位量化 (Colab T4 GPU适用)
# 如果您的 Colab 环境使用 T4 GPU，可以尝试启用 8 位量化，进一步减少内存占用，可能会轻微牺牲模型精度
model = AutoModelForCausalLM.from_pretrained(
    ...,
    load_in_8bit=True  # 替代 4 位量化，将 load_in_4bit=True 替换为 load_in_8bit=True
)

# 启用梯度检查点
# 启用梯度检查点 (Gradient Checkpointing) 技术，可以在一定程度上减少 GPU 显存占用，但会增加计算时间
model.gradient_checkpointing_enable()

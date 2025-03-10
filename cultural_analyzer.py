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

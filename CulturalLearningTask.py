# -*- coding: utf-8 -*-
"""
# 《星辰AI多语言理解系统》Colab实现指南
# 由DeepSeek+Gemini与Win996用户共同设计，适配Google Colab环境
# 本代码基于Hugging Face Transformers库和OpenFL联邦学习框架，
# 旨在构建一个能够理解多语言文化内涵，并生成独特“星辰符号”的AI系统。
# 核心功能包括：文化特征九层解析，星辰符号动态生成，以及联邦学习优化。
"""
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


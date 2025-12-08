# Fourier_Interpreter (demo)
## 模型
本项目现使用 `Qwen3-1.7B`（约1.7B，非量化，中文支持，LLaMA风格的纯解码器结构）。
模型需放置在本地 `model/Qwen3-1.7B` 目录下。
运行时会打印模型的隐藏层数、隐藏维度与注意力头数以供确认。

**版本要求**

- `transformers>=4.51.0`

**显存建议**

- **FP16**：约 4 GB
- **8-bit**：约 2.5 GB
- **4-bit**：约 2 GB

## 测试用例
一个句子对，其内容为**语法(syntax)**或**数学(math)**，每个大类型下面还有多个细分类型。可以在dataset文件中查看所有测试用例。其形式如
{
    "catalog": "syntax",
    "type": "Subject-Verb-Agreement",
    "id": 1,
    "A": "The cat sleeps on the sofa all day.",
    "B": "The cat sleep on the sofa all day.",
    "target_word_A": "sleeps",
    "target_word_B": "sleep"
}
具有以下性质：
1. A是正确的，而B是错误的
2. A和B的token数相同
3. A和B只有一个token不同，这个token标记为target_word。
## 数据处理
将A和B输入模型，测量每个token在每一层的激活向量，并执行FFT，激活值的频谱被保存在 `output/data` 下面。 
将模型每一层执行所有任务产生的激活频谱绘制为可交互页面，用户可以选择单独查看不同类型任务在该层产生的激活频谱。 
正确句子、错误句子和二者差值所对应的频谱其命名前缀分别带有sentence_A、sentence_B和diff。
## 几个有趣的事实
1. 在浅层，语法任务的频谱差值和数学任务没有明显区别；而在深层，语法任务的频谱差值则远大于数学任务。
2. 在浅层和深层，频谱的幅值较小，而在中层，频谱的幅值非常大。这一点对于语法任务更加显著。

## 使用方法

在本地准备好模型目录：

- `model/Qwen3-1.7B/`（包含 `config.json`, `model.safetensors`, `tokenizer.json` 等文件）。

运行管线：

```
python main.py --model_dir Qwen3-1.7B --fast
```

如需下载模型，可运行：

```
python scripts/download_qwen3_1_7b.py
```

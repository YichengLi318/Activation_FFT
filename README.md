## Fourier_Interpreter (demo)
# 模型
GPT-2 large，参数量为774M，隐藏层数为36，hidden dimension为1280。
# 测试用例
形如  
{
    "name": "Subject-Verb-Agreement",
    "A": "The key to the doors is lost.",
    "B": "The key to the doors are lost.",
    "target_word_A": "is",
    "target_word_B": "are"
}，  
保证：
1. A和B的token数相同
2. A和B只有一个token不同，这个token标记为target_word
3. A是正确的，而B是错误的。
# 处理
将A和B输入模型，测量每个token在每一层的激活向量，并执行FFT。将A和B频谱的差值最终绘制3D图：  
x轴：频率  
y轴：层数  
z轴：幅度  
对target_word的激活和句子中除了target_word之外的其他所有词的平均激活同时执行上面的操作。结果储存在output文件夹中。
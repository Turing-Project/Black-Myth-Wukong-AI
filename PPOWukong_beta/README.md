# PPOWukong_Beta

这是一个基于 [EldenRL](https://github.com/ocram444/EldenRL) 的测试项目。

## 运行需求

你需要拥有并安装《黑神话：悟空》（Black Myth: Wukong），并将游戏设置为窗口模式，分辨率为 1920x1080，窗口位于屏幕左上角。在运行任何训练之前，你可以按照以下按键绑定：<code>w,a,s,d = 移动 | shift = 冲刺/闪避 | h = 轻攻击 | j = 重攻击 | r = 治疗 | e = 交互 | esc = 菜单</code>。

为了OCR有较高的精度，需要在标题界面将语言设置为英文。

在正确安装并设置好游戏后，你可以安装代码所需的依赖项。大多数依赖项可以通过简单的 <code>pip install</code> 来安装，但需要注意一些特殊的安装要求：<code>Stable-Baselines3</code> 需要 Python 3.9.13 和 PyTorch 来运行。<code>Pytesseract</code> 也需要下载和安装，并且路径需要在 main.py 中设置，以便从图像中读取文本。

这个项目基于 Windows 11 构建，但在旧版本的 Windows 上也应该可以运行。

除了软件要求外，你还需要硬件来支持训练。该项目已在 CPU 模式下进行测试，游戏正常运行在 GPU 上，训练则运行在 CPU 上。在 R9 8945h CPU 上，训练的测试性能大约为每秒 2帧，游戏正常运行。

## 运行
基本上可以参照原项目[READ.md](https://github.com/ocram444/EldenRL/blob/main/README.md)

不过，需要注意由于黑神话加载界面的不同，检测是否位于加载界面的字段需要随之变化，可以`WukongEnv.py`在中找到。

`walkToBoss.py`的其他Boss可以自行按照调用`pydirectinput`编辑，这个版本给出了从土地庙走到幽魂的处行走路线。

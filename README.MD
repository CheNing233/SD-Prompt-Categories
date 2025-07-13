# 提示词分类工具

## 项目介绍

这是一个基于Gradio的提示词分类工具，主要用于将输入的提示词按照预设的分类规则进行分类。工具会自动将输入内容按逗号分割，并根据`Clothes`、`Others`、`Poses`等文件夹下的`.txt`文件中的关键词进行分类。

## 功能特点

- 可动态添加/删除分类规则
- 分类结果自动去重
- 集成AI分类功能，可将未分类提示词推送给AI进一步分类
- 支持配置OpenAI API Key、Base URL、System Prompt和Model

## 使用说明

1. **安装依赖**
   ```bash
   pip install gradio openai
   ```

2. **运行程序**
   ```bash
   python main.py
   ```

3. **界面操作**
   - 在输入框中输入逗号分隔的提示词
   - 点击"分类"按钮进行分类
   - 点击"保存结果"将分类结果保存到对应文件
   - 在"AI分类配置"中，填写API Key、Base URL、System Prompt和Model，点击"保存AI配置"。
   - 点击"AI分类"按钮，将"未分类"框中的内容发送给AI进行分类，结果将显示在"AI分类结果"框中。

4. **分类管理**
   - 在"分类配置管理"中可以添加/删除分类
   - 添加新分类需要指定分类名称和对应的文件夹路径

## 文件说明

- `extract_*.txt`: 保存分类结果的文本文件
- `config.json`: 包含分类配置信息和AI分类配置（API Key, Base URL, System Prompt, Model）
- `Clothes/`, `Others/`, `Poses/`: 分类关键词目录

## 注意事项

1. 快速保存模式不会检查重复内容
2. 修改分类配置后需手动重启

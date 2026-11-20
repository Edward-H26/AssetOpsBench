# v002

设计：首次仅读取一个传感器组；提示分段计划、完工后验证；仅修改prompt

本版本是已经完成的历史试跑归档，不是重新运行。

- [运行结果](runs/imported/README.md)
- [分析与下一版设计](ANALYSIS.md)

source/prompts.py 是当时使用的完整prompt文件；其余正式代码按 version.json 的 commit 定位。原始LLM输入输出、计划、工具轨迹和评分保存在 runs/imported/ 各模式目录。

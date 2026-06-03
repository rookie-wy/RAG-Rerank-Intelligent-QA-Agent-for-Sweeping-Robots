# RAG-Rerank-Intelligent-QA-Agent-for-Sweeping-Robots

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 项目简介

本项目实现了一个基于RAG（检索增强生成）架构的智能问答智能体，专为扫地机器人知识问答场景设计。系统结合BM25与余弦相似度的混合检索策略，并集成重排模型优化最终结果，能够准确回答关于扫地机器人使用、维护、故障排查等问题。

## 核心特性

-  **混合检索**：结合BM25关键词检索与余弦相似度语义检索，兼顾精确匹配与语义理解
-  **重排优化**：集成重排模型（Rerank Model）对召回结果进行精排，提升答案相关性
-  **智能分块**：通过语义递归切分技术，将文档智能分割为语义完整的文本块
-  **向量存储**：使用Chroma向量数据库存储文本嵌入向量，支持持久化与快速检索
-  **工具调用**：支持通过工具调用完成数据加载、检索、问答等完整流程
-  **智能问答**：基于大语言模型生成准确、自然的回答，支持上下文理解

## 系统架构

```
用户问题 → 混合检索（BM25 + 向量检索）→ 候选片段召回 → 重排模型精排 → LLM生成答案 → 输出结果
```

## 项目结构

```
RAG-Rerank-Intelligent-QA-Agent-for-Sweeping-Robots/
├── agent/               # 智能体核心逻辑
├── chroma_db/          # Chroma向量数据库存储
├── config/             # 配置文件
├── data/               # 原始数据与预处理脚本
├── logs/               # 运行日志
├── prompts/            # 提示词模板
├── pyproject.toml      # 项目依赖配置
└── md5.text           # 数据校验文件
```

## 快速开始

### 环境要求

- Python 3.8+
- pip 或 conda
- 特别提示：可使用uv虚拟项目环境一键配置准确依赖版本

### 安装依赖

```bash
pip install -e .
```

### 配置说明

在 `config/` 目录下配置以下内容：
- API密钥（大语言模型、嵌入模型等）
- 数据库路径
- 检索参数（召回数量、重排阈值等）

### 运行示例

```python
# 加载文档并构建向量数据库
python agent/build_index.py

# 启动问答交互
python agent/qa_agent.py
```

## 技术栈

- **向量数据库**：Chroma
- **嵌入模型**：支持OpenAI Embeddings、HuggingFace Embeddings等
- **检索算法**：BM25 + 余弦相似度
- **重排模型**：Cross-Encoder架构的Rerank模型
- **大语言模型**：支持GPT系列、ChatGLM等
- **文档处理**：语义递归切分、元数据提取

## 应用场景

- 扫地机器人产品说明书问答
- 故障代码解读与维修指导
- 使用技巧与保养建议
- 多型号参数对比查询

## 性能优化

- 混合索引策略平衡关键词与语义匹配
- 两阶段检索（快速召回 + 精准重排）提升效率
- 向量数据库索引优化，支持大规模文档
- 异步处理与缓存机制

## 贡献指南

欢迎提交Issue和Pull Request参与项目改进。

##后续优化方向

-增加“增量索引 + 热更新”
-增加“查询改写”模块
-集成RAGAS评估框架
---

**⭐ 如果这个项目对你有帮助，欢迎给个Star！**

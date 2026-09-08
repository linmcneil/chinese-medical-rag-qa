# 项目路线图（ROADMAP）

定位：把「中文医疗 RAG 问答」做成可复现、可评测、可交付的工程级项目：
数据管线 → 分块 → 检索 → 生成 → 评测 → 发行版，全程脚本化，代码已开源。

## 状态速览（2026-09-08）
- 单元测试 23/23 通过；GitHub Actions CI 已配置
- 检索评测（GPU，1000 问，8524 块知识库）：Hit@1/3/5 = 86.5% / 90.3% / 91.1%，平均 3.5ms/问
- QLoRA 对照实验完成：语义相似度 0.789 → 0.851、Rouge-1 1.25 → 5.46（详见 docs/EXPERIMENT.md）
- 交付：GPU 网页版（app_demo.py）+ 本地 CPU 版（app_local.py）+ CLI；内置样例数据；MIT 开源

## 已完成

### A. 工程标准化
- [x] 包结构：ragqa/{config,data,chunking,prompts,retriever,generator,modelstore,inference,localgen}
- [x] CLI 参数统一；requirements.txt + requirements-cpu.txt 分离；.gitignore 精确放行样例数据
- [x] 模型下载统一走 modelstore.py（镜像优先、延迟触发）
- [x] 单元测试 23 项（纯 CPU、CI 覆盖）
- [~] 日志统一：核心脚本已用 logging，个别演示脚本少量 print（不阻塞）

### B. RAG 质量与评测
- [x] 数据管线：Toyhom 六科室 CSV、蓄水池抽样、与知识库同源可回溯的评测集
- [x] 检索评测真机出数（Hit@1/3/5、延迟、分科室，JSON 输出）
- [x] 生成留出集对比（基座 vs LoRA：semantic 0.789→0.851、Rouge、长度）
- [x] 生成质量工程防线：检索片段清洗 + 相关性门自动降级；去复读/客套/转场截断/限长
- [ ] bge-reranker 重排 vs Top-k 对比（Hit@3 与端到端）
- [ ] 混合检索（BM25 + 向量）或查询改写对比
- [ ] 更大留出集的 LLM-as-judge 生成评测

### C. 演示与交付
- [x] Gradio GPU 网页版（基座/LoRA + RAG 开关）
- [x] Gradio 本地 CPU 版（轻量生成 / 仅检索摘录）
- [x] CLI：REPL / 单问 / 只检索
- [x] README（中/EN）+ 架构图 + 结果表 + 免责声明；release/ 双击 .bat
- [x] GitHub 开源：MIT LICENSE、CI、内置样例数据
- [ ] Dockerfile + docker-compose（可选）
- [ ] 回答附加「命中片段序号引用」的高亮展示（已有片段展示，可再强化）

## 常见设计问题（FAQ）
1. 为什么用 RAG 而不是只微调？
2. 分块大小怎么定、有没有评测？
3. Hit@k 衡量什么、不衡量什么？
4. 检索不到 / 检索跑题怎么办？（已落地：相关性门 + 自动降级）
5. 重复文本怎么处理？
6. 为什么 embedding 选 bge-small-zh？
7. 医疗内容的伦理边界怎么声明？

上述问题的设计思路散落在 README、docs/EXPERIMENT.md、docs/CHANGELOG.md 与对应代码注释中，欢迎在 GitHub Issues 交流。
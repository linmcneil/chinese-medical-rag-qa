# 贡献指南

所有入口调用 `ragqa.factory.create_pipeline`，业务策略集中在核心库；不要在网页或 API 中另写一套提示词或证据规则。模型依赖在后端内部延迟导入。

```bash
python -m pip install -e ".[api,dev]"
python -m pytest -q
python -m ruff check ragqa tests
python -m build
```

修改数据契约需更新 manifest schema 与迁移文档。行为修改应覆盖真实失败路径，包括错误、拒答、输入预算、来源对应关系和训练/验证隔离。测试不依赖远程模型下载，optional integration 测试缺少依赖时应明确 skip。

不覆盖旧数据、旧实验结果或已生成的索引/报告。新效果必须说明数据集、模型版本、配置、硬件、预热和分母；不要把引用编号合法称作答案事实正确。

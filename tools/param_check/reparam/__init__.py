"""tools/param_check/reparam/ — RadarNeXt 模型重参数化（reparameterize）验证。

子模块:
  - model       只算 train vs inference 模式的参数量
  - benchmark   算参数 + 跑 FPS + 验证融合前后数学等价
"""
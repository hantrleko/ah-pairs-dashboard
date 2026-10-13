# A+H 协整与动态配对交易

V0.9。EF Lab 的 A+H 股协整配对交易研究看板。版本记录见 [CHANGELOG.md](CHANGELOG.md)。

A 股和 H 股不能互相兑换。这里的结果是统计套利的历史回测，不是无风险套利。

## 运行

```bash
cd F:\Hermes\Github\ah-pairs-dashboard
pip install -r requirements.txt
streamlit run app.py
```

命令行看一只股票：

```bash
python -m ah_pairs --a 601318 --start 2018-01-01
```

行情缓存在 `data/cache`。股票池快照在 `data/ah_universe.csv`，看板侧栏可以重新从 AKShare 拉取。

## 方法

- 价格用复权收盘价。H 股乘以 `HKDCNY` 后取对数。只保留 A、H 同时开市的日期。
- 全样本 Engle–Granger 和 Johansen 是描述统计，用了窗口内的全部价格。
- 交易用的 β 来自滚动 OLS 或卡尔曼滤波，都只用当天以前的数据。z 分数的均值和标准差同样只用历史价差。
- 默认在信号出现的下一交易日收盘成交，仓位从成交之后的收益开始计算。A 股因此不会在买入当日卖出。页面里可以改成信号当日收盘，那一档更乐观。
- 权重满足 `|w_A| + |w_H| = 1`。β 变动小于再平衡带时不调仓。
- 印花税默认走法定档期：A 股 2023-08-28 起卖方 0.05%，此前按 0.1%；H 股 2021-08-01 至 2023-11-16 为 0.13%，其余 0.10%。
- 北向资金是两市合计净买入，不是个股持仓。2024 年 8 月后日频流向停止披露。回归用 HAC 标准误，只作描述。

## 测试

```bash
python -m pytest tests -q
```

import yfinance as yf
import yfinance_cache as yfc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 获取沪深300ETF日线数据（最近约5年）
df = yf.download('510300.SS', start='2021-01-01', auto_adjust=True, multi_level_index=False, progress=False)

print(df.head())
print(f"\n数据范围: {df.index[0].date()} 到 {df.index[-1].date()}")
print(f"数据总行数: {len(df)}")

# 收盘价走势图
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['Close'], linewidth=1.5)
plt.title('沪深300ETF (510300) 收盘价走势')
plt.xlabel('日期')
plt.ylabel('价格（元）')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

def backtest_dca(df, monthly_amount=1000):
    """
    定投回测 (Dollar Cost Averaging)
    """
    df = df.copy()
    df['month'] = df.index.to_period('M')

    # 每月第一个交易日买入
    monthly_first = df.groupby('month').first()
    shares_bought = monthly_amount / monthly_first['Close']
    total_shares = shares_bought.cumsum()
    total_cost = monthly_amount * np.arange(1, len(monthly_first) + 1)

    # 每月末市值
    monthly_last = df.groupby('month').last()
    portfolio_value = total_shares * monthly_last['Close']
    returns = (portfolio_value.values - total_cost) / total_cost

    result_dca = pd.DataFrame({
        'total_cost': total_cost,
        'total_shares': total_shares.values,
        'portfolio_value': portfolio_value.values,
        'return': returns
    }, index=monthly_first.index)

    return result_dca


result_dca = backtest_dca(df)

print(f"投资期间: {result_dca.index[0]} 到 {result_dca.index[-1]}")
print(f"投资月数: {len(result_dca)}")
print(f"总投入: {result_dca['total_cost'].iloc[-1]:,.0f} 元")
print(f"最终市值: {result_dca['portfolio_value'].iloc[-1]:,.0f} 元")
print(f"最终收益率: {result_dca['return'].iloc[-1]:.2%}")

# 可视化
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ts = result_dca.index.to_timestamp()

ax1 = axes[0]
ax1.plot(ts, result_dca['portfolio_value'], label='持仓市值', linewidth=2)
ax1.plot(ts, result_dca['total_cost'], label='累计成本', linewidth=2, linestyle='--')
ax1.fill_between(ts, result_dca['total_cost'], result_dca['portfolio_value'],
                  where=result_dca['portfolio_value'] > result_dca['total_cost'],
                  alpha=0.3, color='green')
ax1.fill_between(ts, result_dca['total_cost'], result_dca['portfolio_value'],
                  where=result_dca['portfolio_value'] <= result_dca['total_cost'],
                  alpha=0.3, color='red')
ax1.set_ylabel('金额（元）')
ax1.set_title('持仓市值 vs 累计成本')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(ts, result_dca['return'] * 100, linewidth=2)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.fill_between(ts, 0, result_dca['return'] * 100,
                  where=result_dca['return'] > 0, alpha=0.3, color='green')
ax2.fill_between(ts, 0, result_dca['return'] * 100,
                  where=result_dca['return'] <= 0, alpha=0.3, color='red')
ax2.set_ylabel('收益率 (%)')
ax2.set_title('累计收益率')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# STEP3. 基准：一次性买入持有
initial_price = df['Close'].iloc[0]
final_price = df['Close'].iloc[-1]
benchmark_return = (final_price - initial_price) / initial_price

dca_return = result_dca['return'].iloc[-1]

print(f"沪深300ETF 起点价格: {initial_price:.3f}")
print(f"沪深300ETF 终点价格: {final_price:.3f}")
print(f"\n一次性买入收益率: {benchmark_return:.2%}")
print(f"定投收益率: {dca_return:.2%}")
print(f"差异: {dca_return - benchmark_return:.2%}")

if dca_return > benchmark_return:
    print("\n→ 定投跑赢了一次性买入！")
    print("  在下跌市中，定投通过分批买入避开了部分高位。")
else:
    print("\n→ 定投跑输了一次性买入！")
    print("  在上涨市中，越早全部投入越好。")

print("\n但两种策略都没有 Alpha——收益完全取决于市场涨跌，这就是 Beta。")

# STEP4: 对比图：定投 vs 买入持有
df_monthly = df.copy()
df_monthly['month'] = df_monthly.index.to_period('M')
monthly_last = df_monthly.groupby('month').last()
bnh_monthly_return = (monthly_last['Close'].values - initial_price) / initial_price

fig, ax = plt.subplots(figsize=(12, 6))
ts = result_dca.index.to_timestamp()
ax.plot(ts, result_dca['return'] * 100, label='定投', linewidth=2)
ax.plot(ts, bnh_monthly_return * 100, label='一次性买入持有', linewidth=2, linestyle='--')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('累计收益率 (%)')
ax.set_title('定投 vs 一次性买入持有')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 取最近 1 年数据，让图更清晰
one_year_ago = df.index[-1] - pd.DateOffset(years=1)
df_recent = df.loc[one_year_ago:].copy()

# 计算三条均线
ma10 = df_recent['Close'].rolling(10).mean()
ma20 = df_recent['Close'].rolling(20).mean()
ma60 = df_recent['Close'].rolling(60).mean()

# 找出收盘价与 MA20 的交叉点
cross_up = (df_recent['Close'].shift(1) <= ma20.shift(1)) & (df_recent['Close'] > ma20)
cross_down = (df_recent['Close'].shift(1) >= ma20.shift(1)) & (df_recent['Close'] < ma20)

# 画图
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(df_recent.index, df_recent['Close'], color='black', linewidth=1.5, label='收盘价')
ax.plot(df_recent.index, ma10, color='green', linewidth=1, linestyle='--', label='10日均线')
ax.plot(df_recent.index, ma20, color='#1f77b4', linewidth=1.5, linestyle='-', label='20日均线')
ax.plot(df_recent.index, ma60, color='red', linewidth=1, linestyle=':', label='60日均线')

# 标记交叉点
ax.scatter(df_recent.index[cross_up], df_recent['Close'][cross_up],
           marker='^', color='green', s=100, zorder=5, label='上穿MA20（买入信号）')
ax.scatter(df_recent.index[cross_down], df_recent['Close'][cross_down],
           marker='v', color='red', s=100, zorder=5, label='下穿MA20（卖出信号）')

ax.set_title('沪深300ETF：收盘价与均线（最近1年）')
ax.set_xlabel('日期')
ax.set_ylabel('价格（元）')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 交叉统计
print(f"\nMA20 交叉统计（最近1年）：")
print(f"  上穿（买入信号）：{cross_up.sum()} 次")
print(f"  下穿（卖出信号）：{cross_down.sum()} 次")

# STEP5
def backtest_dca_with_ma(df, monthly_amount=1000, ma_period=20):
    """
    带均线条件的定投回测
    规则：每月第一个交易日，如果收盘价 > N日均线，则买入；否则跳过
    """
    df = df.copy()
    df['ma'] = df['Close'].rolling(ma_period).mean()
    df['month'] = df.index.to_period('M')
    monthly_first = df.groupby('month').first()
    monthly_first['signal'] = monthly_first['Close'] > monthly_first['ma']

    shares_list, cost_list = [], []
    total_shares, total_cost = 0, 0

    for idx, row in monthly_first.iterrows():
        if pd.notna(row['ma']) and row['signal']:
            total_shares += monthly_amount / row['Close']
            total_cost += monthly_amount
        shares_list.append(total_shares)
        cost_list.append(total_cost)

    monthly_last = df.groupby('month').last()
    portfolio_value = np.array(shares_list) * monthly_last['Close'].values
    cost_array = np.array(cost_list)
    returns = np.zeros(len(cost_array))
    mask = cost_array > 0
    returns[mask] = (portfolio_value[mask] - cost_array[mask]) / cost_array[mask]

    return pd.DataFrame({
        'total_cost': cost_list,
        'total_shares': shares_list,
        'portfolio_value': portfolio_value,
        'return': returns,
        'signal': monthly_first['signal'].values
    }, index=monthly_first.index)


# 对比不同均线参数
print("均线择时策略：不同参数对比")
print("=" * 45)

ma_results = {}
for ma in [10, 20, 60]:
    result = backtest_dca_with_ma(df, ma_period=ma)
    ma_results[f'MA{ma}'] = result
    buy_count = int(result['signal'].sum())
    total_months = len(result)
    print(f"MA{ma:3d}: 收益率 {result['return'].iloc[-1]:7.2%}, 买入 {buy_count:2d}/{total_months} 次")

print(f"\n简单定投: 收益率 {result_dca['return'].iloc[-1]:.2%}")

# 对比图
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(result_dca.index.to_timestamp(), result_dca['return'] * 100,
        label='简单定投', linewidth=2, linestyle='--', color='gray')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for (name, result), color in zip(ma_results.items(), colors):
    ax.plot(result.index.to_timestamp(), result['return'] * 100,
            label=name, linewidth=2, color=color)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('收益率 (%)')
ax.set_title('均线择时 vs 简单定投')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 参数扫描
scan_results = []
for ma in range(5, 121, 5):
    result = backtest_dca_with_ma(df, ma_period=ma)
    scan_results.append({'ma_period': ma, 'return': result['return'].iloc[-1]})

scan_df = pd.DataFrame(scan_results)
best = scan_df.loc[scan_df['return'].idxmax()]

print(f"最优参数: MA{int(best['ma_period'])}")
print(f"最优收益率: {best['return']:.2%}")
print(f"简单定投收益率: {result_dca['return'].iloc[-1]:.2%}")
print(f"收益提升: {best['return'] - result_dca['return'].iloc[-1]:.2%}")

# 柱状图
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['green' if r > 0 else 'red' for r in scan_df['return']]
ax.bar(scan_df['ma_period'], scan_df['return'] * 100, color=colors, alpha=0.7)
ax.bar(best['ma_period'], best['return'] * 100, color='gold', edgecolor='black', linewidth=2,
       label=f"最优 MA{int(best['ma_period'])}")
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axhline(y=result_dca['return'].iloc[-1] * 100, color='gray', linestyle='--', linewidth=2,
           label='简单定投')
ax.set_xlabel('均线周期（天）')
ax.set_ylabel('收益率 (%)')
ax.set_title('参数扫描：不同均线周期的收益率')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 样本外测试
split_idx = int(len(df) * 0.6)
split_date = df.index[split_idx]
train = df.loc[:split_date]
test = df.loc[split_date:]

print(f"训练集: {train.index[0].date()} 到 {train.index[-1].date()} ({len(train)} 条)")
print(f"测试集: {test.index[0].date()} 到 {test.index[-1].date()} ({len(test)} 条)")

# 训练集上找最优参数
train_scan = []
for ma in range(5, 121, 5):
    result = backtest_dca_with_ma(train, ma_period=ma)
    train_scan.append({'ma': ma, 'return': result['return'].iloc[-1]})
train_scan_df = pd.DataFrame(train_scan)
best_ma = int(train_scan_df.loc[train_scan_df['return'].idxmax()]['ma'])
best_train_return = train_scan_df['return'].max()

# 测试集上验证
test_ma_result = backtest_dca_with_ma(test, ma_period=best_ma)
test_ma_return = test_ma_result['return'].iloc[-1]

test_simple_result = backtest_dca(test)
test_simple_return = test_simple_result['return'].iloc[-1]

print(f"\n训练集最优: MA{best_ma}, 收益率 {best_train_return:.2%}")
print(f"测试集表现: MA{best_ma}, 收益率 {test_ma_return:.2%}")
print(f"测试集简单定投: 收益率 {test_simple_return:.2%}")

print("\n" + "=" * 50)
if test_ma_return < test_simple_return:
    print("这就是过拟合——你在历史数据上找到的是巧合，不是规律。")
else:
    print("结果看起来还行，但仍需谨慎，样本量有限。")
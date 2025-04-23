import matplotlib.pyplot as plt
import numpy as np

# 数据准备
versions = ['V1\nnaive', 'V2\n编译优化', 'V3\n共享内存优化', 'V4\nWarp优化', 'V5\n矩阵优化', 'V6\n向量化']
avg_perf = [837.26, 855.90, 1514.41, 2032.73, 3303.43, 4522.41]
max_perf = [1590, 1590, 2550, 2550, 5950, 7600]

# 创建图形
plt.figure(figsize=(13, 7))

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置位置
x = np.arange(len(versions))

# 绘制折线图
plt.plot(x, avg_perf, 'o-', label='平均性能', color='#483D8B', linewidth=2, markersize=8)
plt.plot(x, max_perf, 'o-', label='最大性能', color='#8B4513', linewidth=2, markersize=8)

# 添加数值标签
for i, (avg, max_val) in enumerate(zip(avg_perf, max_perf)):
    plt.annotate(f'{avg:.1f}', 
                (i, avg),
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center',
                color='#483D8B',
                fontweight='bold')
    plt.annotate(f'{max_val:.1f}', 
                (i, max_val),
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center',
                color='#8B4513',
                fontweight='bold')

# 添加性能提升的百分比标注
for i in range(1, len(avg_perf)):
    avg_increase = (avg_perf[i] - avg_perf[i-1]) / avg_perf[i-1] * 100
    max_increase = (max_perf[i] - max_perf[i-1]) / max_perf[i-1] * 100
    
    # 平均性能增长百分比
    plt.annotate(f'+{avg_increase:.1f}%',
                xy=((i + i-1)/2, (avg_perf[i] + avg_perf[i-1])/2),
                xytext=(0, -20),
                textcoords='offset points',
                ha='center',
                va='top',
                color='#483D8B',
                fontweight='bold')
    
    # 最大性能增长百分比
    plt.annotate(f'+{max_increase:.1f}%',
                xy=((i + i-1)/2, (max_perf[i] + max_perf[i-1])/2),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                va='bottom',
                color='#8B4513',
                fontweight='bold')

# 设置标题和标签
plt.title('不同版本的性能对比', fontsize=14, pad=20)
plt.xlabel('优化版本', fontsize=12)
plt.ylabel('性能 (Gflops/s)', fontsize=12)

# 设置x轴刻度
plt.xticks(x, versions)

# 添加图例
plt.legend(loc='upper left')

# 设置y轴范围
plt.ylim(0, max(max_perf) * 1.1)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('result2.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
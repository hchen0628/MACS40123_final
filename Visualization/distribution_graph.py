import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
final_df = pd.read_csv(r"C:\Users\A1157\Downloads\expression_patterns_dataset.csv")

# Set scientific paper style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Calculate data for each period
period_data = []
for period in range(1, 6):
    period_df = final_df[final_df['period'] == period]
    total = len(period_df)
    
    # Calculate specific counts for each pattern
    data = {
        'Total': total,
        'Direct Non-critical (0,0)': len(period_df[(period_df['criticism'] == 0) & (period_df['vague'] == 0)]),
        'Vague Non-critical (0,1)': len(period_df[(period_df['criticism'] == 0) & (period_df['vague'] == 1)]),
        'Direct Critical (1,0)': len(period_df[(period_df['criticism'] == 1) & (period_df['vague'] == 0)]),
        'Vague Critical (1,1)': len(period_df[(period_df['criticism'] == 1) & (period_df['vague'] == 1)])
    }
    period_data.append(data)

# Convert to DataFrame
plot_df = pd.DataFrame(period_data)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Set x-axis positions
x = np.arange(5)
period_labels = ['2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023']

# Set colors
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
patterns = ['Direct Non-critical (0,0)', 'Vague Non-critical (0,1)', 
            'Direct Critical (1,0)', 'Vague Critical (1,1)']

# Draw stacked bar chart
bottom = np.zeros(5)
bars = []
for pattern, color in zip(patterns, colors):
    bars.append(ax.bar(x, plot_df[pattern], bottom=bottom, label=pattern, color=color))
    bottom += plot_df[pattern]

# Add title and labels
ax.set_title('Post Volume and Expression Pattern Distribution\nAcross Five Periods (2018-2023)', 
             pad=20, fontsize=14, fontweight='bold')
ax.set_xlabel('Time Periods', fontsize=12, labelpad=10)
ax.set_ylabel('Number of Posts', fontsize=12, labelpad=10)

# Set x-axis ticks
ax.set_xticks(x)
ax.set_xticklabels(period_labels, rotation=0)

# Add value labels on bars
def add_value_labels(ax, bars):
    for i, bar in enumerate(bars):
        for j, rect in enumerate(bar):
            height = rect.get_height()
            percentage = (height / plot_df['Total'].iloc[j]) * 100
            if percentage >= 3:  # Only show labels for percentages >= 3% to avoid overcrowding
                ax.text(rect.get_x() + rect.get_width()/2., 
                       rect.get_y() + height/2.,
                       f'{percentage:.1f}%',
                       ha='center', va='center',
                       fontsize=8, color='white',
                       fontweight='bold')

add_value_labels(ax, bars)

# Add total counts at top of bars
for i in range(len(x)):
    total = plot_df['Total'].iloc[i]
    ax.text(i, total, f'{total:,}',
            ha='center', va='bottom',
            fontsize=10)

# Add grid lines
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# Add legend
ax.legend(bbox_to_anchor=(1.02, 1), 
         loc='upper left',
         borderaxespad=0.,
         frameon=True,
         fontsize=8,
         handlelength=1.0,
         handletextpad=0.5,
         labelspacing=0.5,
         columnspacing=1.0)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('posts_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistics
print("\nDetailed statistics for each period:")
for period in range(5):
    print(f"\nPeriod {period+1} ({period_labels[period]}):")
    print(f"Total posts: {plot_df['Total'].iloc[period]:,}")
    for pattern in patterns:
        count = plot_df[pattern].iloc[period]
        percentage = (count / plot_df['Total'].iloc[period]) * 100
        print(f"{pattern}: {count:,} ({percentage:.1f}%)")
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'outputs', 'model_metrics.csv')
images_dir = os.path.join(base_dir, '..', 'Images')

# Load metrics
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Metrics file not found at: {csv_path}")

df = pd.read_csv(csv_path)

# Map model names to cleaner display versions
model_map = {
    'LogisticRegression': 'Logistic Regression',
    'DecisionTree': 'Decision Tree',
    'KNN': 'KNN',
    'RandomForest': 'Random Forest'
}
df['Model'] = df['Model'].map(model_map)

# Select target metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
df_filtered = df[['Model'] + metrics].copy()
df_filtered.rename(columns={'F1-Score': 'F1 Score'}, inplace=True)

# Print comparison table
print("="*60)
print("                  MODEL COMPARISON TABLE                    ")
print("="*60)
print(df_filtered.to_string(index=False))
print("="*60)

# Melt data for plotting
df_melted = pd.melt(df_filtered, id_vars=['Model'], var_name='Metric', value_name='Score')

# Plot setup
plt.figure(figsize=(11, 6.5))
sns.set_theme(style="whitegrid")

# Custom premium colors
colors = ["#4F46E5", "#0EA5E9", "#10B981", "#F59E0B"] # Indigo, Sky, Emerald, Amber
sns.set_palette(sns.color_palette(colors))

# Create bar chart
ax = sns.barplot(
    data=df_melted,
    x='Metric',
    y='Score',
    hue='Model',
    edgecolor='gray',
    linewidth=0.8
)

# Customize title and labels
plt.title('Performance Metrics Comparison across Models', fontsize=15, fontweight='bold', pad=18)
plt.ylabel('Score Value', fontsize=12, fontweight='bold', labelpad=10)
plt.xlabel('Evaluation Metrics', fontsize=12, fontweight='bold', labelpad=10)
plt.ylim([0, 1.15]) # Room for bar annotations

# Add score annotations on top of each bar
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.3f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points', fontsize=9, fontweight='semibold')

# Premium Legend
plt.legend(title='Model Type', title_fontsize='11', loc='upper right', frameon=True, shadow=False)

# Adjust layout
plt.tight_layout()

# Save to Images directory
os.makedirs(images_dir, exist_ok=True)
output_path_images = os.path.join(images_dir, 'accuracy_comparison.png')
plt.savefig(output_path_images, dpi=300)

# Save to root workspace directory as well
output_path_root = os.path.join(base_dir, '..', 'accuracy_comparison.png')
plt.savefig(output_path_root, dpi=300)

print(f"Comparison chart successfully generated and saved to:")
print(f"1. {output_path_images}")
print(f"2. {output_path_root}")
print("="*60)

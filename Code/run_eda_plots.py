import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# Directory
images_dir = 'Images'
os.makedirs(images_dir, exist_ok=True)

# Load data
df = pd.read_csv('Datasets/heart.csv')

# 1. Target Class Distribution
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df, x='target', palette='Set2')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontweight='bold')
plt.title('Distribution of Heart Disease Target Class', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Diagnosis Target (0 = Healthy, 1 = Heart Disease)', fontsize=11)
plt.ylabel('Patient Count', fontsize=11)
plt.ylim(0, df['target'].value_counts().max() * 1.15)
plt.savefig(os.path.join(images_dir, 'eda_target_distribution.png'), bbox_inches='tight', dpi=300)
plt.close()

# 2. Age Distribution
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x='age', hue='target', kde=True, palette='muted', multiple='stack', bins=20)
plt.title('Age Distribution vs Heart Disease Diagnosis', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Patient Age (Years)', fontsize=11)
plt.ylabel('Patient Count', fontsize=11)
plt.savefig(os.path.join(images_dir, 'eda_age_distribution.png'), bbox_inches='tight', dpi=300)
plt.close()

# 3. Gender Distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='sex', hue='target', palette='pastel')
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
plt.title('Heart Disease Prevalence by Gender', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Biological Gender (0 = Female, 1 = Male)', fontsize=11)
plt.ylabel('Patient Count', fontsize=11)
plt.ylim(0, df['sex'].value_counts().max() * 1.15)
plt.savefig(os.path.join(images_dir, 'eda_gender_distribution.png'), bbox_inches='tight', dpi=300)
plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(11, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True, 
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of Heart Disease Features', fontsize=14, fontweight='bold', pad=15)
plt.savefig(os.path.join(images_dir, 'eda_correlation_heatmap.png'), bbox_inches='tight', dpi=300)
plt.close()

# 5. Distribution Plots (Chest pain & Heart rate)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(data=df, x='cp', hue='target', ax=axes[0], palette='Set2')
axes[0].set_title('Heart Disease Status by Chest Pain Type (cp)', fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel('Chest Pain Type', fontsize=10)
axes[0].set_ylabel('Patient Count', fontsize=10)

sns.kdeplot(data=df, x='thalach', hue='target', fill=True, ax=axes[1], palette='crest', alpha=0.5, common_norm=False)
axes[1].set_title('Max Heart Rate Achieved (thalach) Density vs Target', fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel('Max Heart Rate Achieved (bpm)', fontsize=10)
axes[1].set_ylabel('Density Estimation', fontsize=10)

plt.savefig(os.path.join(images_dir, 'eda_heart_disease_distribution_plots.png'), bbox_inches='tight', dpi=300)
plt.close()

print("All EDA PNG graphs saved successfully under Images/ directory.")

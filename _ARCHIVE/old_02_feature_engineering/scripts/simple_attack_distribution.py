import matplotlib.pyplot as plt
import pandas as pd
import os

print("Loading UNSW-NB15 training dataset...")

# Load the dataset
df = pd.read_csv('../datasetsfinalproject/UNSW_NB15_training-set.csv')
print(f"Dataset loaded: {len(df)} records")

# Get attack distribution
attack_counts = df['attack_cat'].value_counts().sort_values(ascending=False)
print(f"Attack types found: {len(attack_counts)}")

# Create the graph
plt.figure(figsize=(14, 8))
bars = plt.bar(range(len(attack_counts)), attack_counts.values, alpha=0.8, edgecolor='black')

# Customize
plt.title('Attack Types Distribution - Original UNSW-NB15 Dataset', fontsize=16, fontweight='bold')
plt.xlabel('Attack Types', fontsize=14)
plt.ylabel('Number of Records', fontsize=14)
plt.xticks(range(len(attack_counts)), attack_counts.index, rotation=45, ha='right')

# Add value labels
for i, (bar, count) in enumerate(zip(bars, attack_counts.values)):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(attack_counts.values) * 0.01,
             f'{count:,}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

# Save the graph
if not os.path.exists('results'):
    os.makedirs('results')
    
plt.savefig('results/attack_distribution_statistics.png', dpi=300, bbox_inches='tight')
print("Statistical graph saved: results/attack_distribution_statistics.png")

# Print summary
print("\nAttack Distribution Summary:")
print("=" * 40)
total = len(df)
for attack_type, count in attack_counts.items():
    percentage = (count / total) * 100
    print(f"{attack_type:15}: {count:6,} records ({percentage:5.1f}%)")

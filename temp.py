import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Set a professional style similar to the templates
sns.set_theme(style="whitegrid")

# --- 1. DATA LOADING & GENERATION ---

def generate_mock_data(n_samples=200):
    """
    Generates two datasets that mimic the distributions in the templates.
    Data 1: Lower mean (~72), higher variance.
    Data 2: Higher mean (~87), lower variance.
    """
    # Data 1 Simulation
    data1_acc = np.random.normal(loc=72.3, scale=9.9, size=n_samples)
    data1_acc = np.clip(data1_acc, 40, 95)
    
    # Data 2 Simulation (Correlated with Data 1 to simulate "Same Run")
    # We add a boost to Data 1 to simulate the improved activation on the same seed
    data2_acc = data1_acc + np.random.normal(loc=15, scale=5, size=n_samples)
    data2_acc = np.clip(data2_acc, 70, 98) 
    
    df1 = pd.DataFrame({'Final_Accuracy': data1_acc})
    df1['Run_Index'] = np.arange(len(df1))
    
    df2 = pd.DataFrame({'Final_Accuracy': data2_acc})
    df2['Run_Index'] = np.arange(len(df2))
    
    return df1, df2

def load_data_from_csv(file_path):
    """
    Loads data assuming the structure: 
    ID, Act1, Act2, Loss1...Loss10, Final_Accuracy
    """
    try:
        df = pd.read_csv(file_path, header=None)
        
        column_names = ['ID', 'Act1', 'Act2'] + \
                       [f'Loss_{i}' for i in range(1, 11)] + \
                       ['Final_Accuracy']
        
        if len(df.columns) == 14:
            df.columns = column_names
        else:
            print(f"Warning: {file_path} has {len(df.columns)} columns. Using last col as Accuracy.")
            df.rename(columns={df.columns[-1]: 'Final_Accuracy'}, inplace=True)
            
        df['Run_Index'] = df.index
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# --- SWITCH: Load your real CSV files here ---
file_1_path = '/Users/khaos/Desktop/main-main/results_sorted_nodup.csv' 
file_2_path = '/Users/khaos/Desktop/main-main/results_sorted_tanh_relu.csv'

print(f"Attempting to load {file_1_path} and {file_2_path}...")
df1 = load_data_from_csv(file_1_path)
df2 = load_data_from_csv(file_2_path)

if df1 is None or df2 is None:
    print("Could not load one or both CSV files. Generating mock data...")
    df1, df2 = generate_mock_data(300)
else:
    # Ensure both dataframes have the same length for paired plotting
    min_len = min(len(df1), len(df2))
    df1 = df1.iloc[:min_len]
    df2 = df2.iloc[:min_len]
    print("Successfully loaded and aligned CSV data.")


# --- 2. PLOTTING FUNCTIONS ---

def plot_side_by_side_histograms(d1, d2):
    """Replicates the 'Data 1' vs 'Data 2' side-by-side histogram view."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    
    datasets = [(d1, "Data 1", axes[0]), (d2, "Data 2", axes[1])]
    
    for df, label, ax in datasets:
        acc = df['Final_Accuracy']
        mu, std = stats.norm.fit(acc)
        ax.hist(acc, bins=20, density=True, alpha=0.7, color='#1f77b4', edgecolor='white', label=label)
        
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2, color='#ff7f0e', label=f'N({mu:.2f}, {std:.2f}²)')
        
        ax.set_title(f"{label} Distribution\nMean: {mu:.2f}", fontsize=14)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Accuracy Distribution Analysis", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('1_side_by_side_histograms.png')
    print("Generated: 1_side_by_side_histograms.png")

def plot_violin_comparison(d1, d2):
    """Replicates the 'Accuracy Violin Plot Comparison'."""
    plt.figure(figsize=(10, 6))
    data_to_plot = [d1['Final_Accuracy'], d2['Final_Accuracy']]
    parts = plt.violinplot(data_to_plot, showmeans=True, showmedians=True, showextrema=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.6)
    
    plt.xticks([1, 2], ['Data 1', 'Data 2'], fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy Violin Plot Comparison', fontsize=14, weight='bold')
    plt.grid(True, axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig('2_violin_comparison.png')
    print("Generated: 2_violin_comparison.png")

def plot_sequence_view(d1, d2):
    """Replicates the 'Run-wise Accuracy (Sequence View)'."""
    plt.figure(figsize=(14, 6))
    limit = 200
    y1 = d1['Final_Accuracy'].iloc[:limit]
    y2 = d2['Final_Accuracy'].iloc[:limit]
    x = np.arange(len(y1))
    
    plt.plot(x, y1, marker='o', markersize=5, linestyle='-', linewidth=1, label='Data 1', color='#1f77b4', alpha=0.8)
    plt.plot(x, y2, marker='x', markersize=5, linestyle='-', linewidth=1, label='Data 2', color='#ff7f0e', alpha=0.8)
    
    plt.title('Run-wise Accuracy (Sequence View)', fontsize=14, weight='bold')
    plt.xlabel('Run Index')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('3_sequence_view.png')
    print("Generated: 3_sequence_view.png")

def plot_overlay_histogram(d1, d2):
    """Replicates the 'Accuracy Distributions Comparison' (Overlaid)."""
    plt.figure(figsize=(12, 7))
    acc1 = d1['Final_Accuracy']
    acc2 = d2['Final_Accuracy']
    
    plt.hist(acc1, bins=25, density=True, alpha=0.5, label='Data 1', color='#1f77b4')
    plt.hist(acc2, bins=25, density=True, alpha=0.5, label='Data 2', color='#ff7f0e')
    
    plt.title('Accuracy Distributions Comparison', fontsize=14, weight='bold')
    plt.xlabel('Accuracy')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('4_overlay_distributions.png')
    print("Generated: 4_overlay_distributions.png")

def plot_paired_scatter(d1, d2):
    """
    NEW: Plots a paired scatter plot comparing 'Run X' in File 1 vs 'Run X' in File 2.
    Demonstrates specific row-to-row correspondence.
    """
    plt.figure(figsize=(10, 10))
    
    acc1 = d1['Final_Accuracy']
    acc2 = d2['Final_Accuracy']
    
    # Logic to color code points: Green if File 2 > File 1, Red otherwise
    colors = np.where(acc2 > acc1, '#2ca02c', '#d62728')
    
    # Scatter plot
    plt.scatter(acc1, acc2, c=colors, alpha=0.6, edgecolors='w', s=60, label='Single Run Result')
    
    # Identity Line (x=y)
    # This line represents "No Change". Points above it = Improvement.
    min_val = min(acc1.min(), acc2.min()) - 1
    max_val = max(acc1.max(), acc2.max()) + 1
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='No Change Line (y=x)')
    
    plt.title('Paired Performance: File 1 vs File 2 (Per Run)', fontsize=16, weight='bold')
    plt.xlabel('Accuracy (File 1 / Baseline)', fontsize=14)
    plt.ylabel('Accuracy (File 2 / New Activation)', fontsize=14)
    
    # Statistics
    better_count = np.sum(acc2 > acc1)
    total = len(acc1)
    plt.text(min_val + 1, max_val - 2, 
             f"Improved: {better_count}/{total} ({better_count/total:.1%}) runs", 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('5_paired_scatter.png')
    print("Generated: 5_paired_scatter.png")

# --- 3. EXECUTION ---
print("Generating plots based on the templates...")
plot_side_by_side_histograms(df1, df2)
plot_violin_comparison(df1, df2)
plot_sequence_view(df1, df2)
plot_overlay_histogram(df1, df2)
plot_paired_scatter(df1, df2) # The new plot
print("Done.")
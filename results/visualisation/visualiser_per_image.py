import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec

def load_and_prepare_data(json_file_path):
    """Load evaluation data from JSON and convert to DataFrame"""
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Flatten the nested structure
    rows = []
    for image_name, evaluations in data.items():
        for evaluation in evaluations:
            row = {
                'image_name': image_name,
                **evaluation,
                'error': abs(evaluation['prediction'] - evaluation['ground_truth']),
                'relative_error': abs(evaluation['prediction'] - evaluation['ground_truth']) / 
                                 max(evaluation['ground_truth'], 1e-10) * 100  # Avoid division by zero
            }
            rows.append(row)
    
    return pd.DataFrame(rows)

def visualize_image_evaluations(df, output_dir='evaluation_visualizations'):
    """Create visualizations for each image"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique images
    unique_images = df['image_name'].unique()
    
    for image_name in unique_images:
        image_df = df[df['image_name'] == image_name]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig)
        
        # Comparison of predictions vs ground truth
        ax1 = fig.add_subplot(gs[0, :2])
        pivot_df = image_df.pivot(index='preprocessing', columns='model', values='prediction')
        pivot_df.plot(kind='bar', ax=ax1)
        ax1.axhline(y=image_df['ground_truth'].iloc[0], color='r', linestyle='-', linewidth=2)
        ax1.text(0, image_df['ground_truth'].iloc[0]*1.05, f"Ground Truth: {image_df['ground_truth'].iloc[0]}", 
                color='r', fontweight='bold')
        ax1.set_title(f'Predictions vs Ground Truth for {image_name}')
        ax1.set_ylabel('Value')
        ax1.tick_params(axis='x', rotation=45)
        
        # Absolute error by model and preprocessing method
        ax2 = fig.add_subplot(gs[0, 2:])
        sns.heatmap(image_df.pivot(index='preprocessing', columns='model', values='error'),
                   annot=True, cmap='YlOrRd', ax=ax2)
        ax2.set_title('Absolute Error')
        
        # Relative error by model and preprocessing
        ax3 = fig.add_subplot(gs[1, :2])
        sns.heatmap(image_df.pivot(index='preprocessing', columns='model', values='relative_error'),
                   annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3)
        ax3.set_title('Relative Error (%)')
        
        # Best performing combinations
        ax4 = fig.add_subplot(gs[1, 2:])
        
        # Sort by error
        best_df = image_df.sort_values('error')
        # Plot top 5 or all if less than 5
        n_best = min(5, len(best_df))
        best_combinations = best_df.iloc[:n_best][['preprocessing', 'model', 'prediction', 'ground_truth', 'error']]
        
        # Create a table
        cell_text = []
        for _, row in best_combinations.iterrows():
            cell_text.append([
                row['preprocessing'], 
                row['model'], 
                f"{row['prediction']:.2f}", 
                f"{row['ground_truth']:.2f}", 
                f"{row['error']:.2f}"
            ])
        
        ax4.axis('off')
        table = ax4.table(
            cellText=cell_text,
            colLabels=['Preprocessing', 'Model', 'Pred', 'Truth', 'Error'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax4.set_title('Best Performing Combinations')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{image_name.split('.')[0]}_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_visualization(df, output_dir='evaluation_visualizations'):
    """Create overall summary visualizations across all images"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate aggregate statistics
    model_preprocessing_stats = df.groupby(['model', 'preprocessing']).agg({
        'error': ['mean', 'median', 'std'],
        'relative_error': ['mean', 'median', 'std']
    }).reset_index()
    
    # Flatten the column names
    model_preprocessing_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in model_preprocessing_stats.columns.values]
    
    # Create figure for summary visualizations
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(3, 2, figure=fig)
    
    # Average error by model and preprocessing
    ax1 = fig.add_subplot(gs[0, 0])
    pivoted = df.pivot_table(values='error', index='preprocessing', columns='model', aggfunc='mean')
    sns.heatmap(pivoted, annot=True, cmap='YlOrRd', ax=ax1)
    ax1.set_title('Average Absolute Error by Model and Preprocessing')
    
    # Average relative error by model and preprocessing
    ax2 = fig.add_subplot(gs[0, 1])
    pivoted = df.pivot_table(values='relative_error', index='preprocessing', columns='model', aggfunc='mean')
    sns.heatmap(pivoted, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2)
    ax2.set_title('Average Relative Error (%) by Model and Preprocessing')
    
    # Box plot of errors by model
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(x='model', y='error', data=df, ax=ax3)
    ax3.set_title('Distribution of Absolute Errors by Model')
    ax3.tick_params(axis='x', rotation=45)
    
    # Box plot of errors by preprocessing
    ax4 = fig.add_subplot(gs[1, 1])
    sns.boxplot(x='preprocessing', y='error', data=df, ax=ax4)
    ax4.set_title('Distribution of Absolute Errors by Preprocessing')
    ax4.tick_params(axis='x', rotation=45)
    
    # Overall best performing combinations
    ax5 = fig.add_subplot(gs[2, :])
    # Group by model and preprocessing and calculate mean errors
    best_combinations = df.groupby(['model', 'preprocessing']).agg({
        'error': 'mean',
        'relative_error': 'mean'
    }).reset_index().sort_values('error')
    
    # Top 10 combinations
    top_10 = best_combinations.head(10)
    
    y_pos = np.arange(len(top_10))
    labels = [f"{row['preprocessing']} + {row['model']}" for _, row in top_10.iterrows()]
    
    # Create horizontal bar chart
    bars = ax5.barh(y_pos, top_10['error'], align='center')
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(labels)
    ax5.invert_yaxis()  # Labels read top-to-bottom
    ax5.set_xlabel('Average Absolute Error')
    ax5.set_title('Top 10 Model-Preprocessing Combinations by Average Error')
    
    # Add error values as text
    for i, v in enumerate(top_10['error']):
        ax5.text(v + 0.1, i, f"{v:.2f}", va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

def run_visualization(json_file_path='image_results.json', output_dir='evaluation_visualizations'):
    """Run the complete visualization pipeline"""
    print("Loading and preparing data...")
    df = load_and_prepare_data(json_file_path)
    
    print(f"Creating individual visualizations for {df['image_name'].nunique()} images...")
    visualize_image_evaluations(df, output_dir)
    
    print("Creating summary visualizations...")
    create_summary_visualization(df, output_dir)
    
    print(f"Visualizations complete. Output saved to {output_dir}/")

# Example usage
if __name__ == "__main__":
    run_visualization()
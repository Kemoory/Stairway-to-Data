import os
import numpy as np
import cv2
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Create a custom colormap for better visualization
custom_cmap = LinearSegmentedColormap.from_list('custom_YlOrRd', 
                                            ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', 
                                             '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026'])

# Feature extraction and dataset preparation
def extract_features(image_path):
    """
    Extract features from an image for stair counting.
    Returns a feature vector for the given image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 200))
    edges = cv2.Canny(resized, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    horizontal_line_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 20 or angle > 160:
                horizontal_line_count += 1
    
    win_size = (200, 200)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(resized)
    
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    gradient_x_mean = np.mean(np.abs(sobelx))
    gradient_y_mean = np.mean(np.abs(sobely))
    
    custom_features = np.array([
        horizontal_line_count,
        edge_density,
        gradient_x_mean,
        gradient_y_mean
    ])
    
    hog_features_reduced = hog_features[::20].flatten()
    all_features = np.concatenate([custom_features, hog_features_reduced])
    
    return all_features

def prepare_dataset(image_paths, labels):
    """
    Process all images and prepare feature vectors and labels.
    """
    features_list = []
    valid_labels = []
    valid_paths = []
    
    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(image_paths)}")
        
        features = extract_features(image_path)
        if features is not None:
            features_list.append(features)
            valid_labels.append(label)
            valid_paths.append(image_path)
    
    if not features_list:
        return np.array([]), np.array([]), []
    
    max_length = max(len(f) for f in features_list)
    standardized_features = []
    for f in features_list:
        if len(f) < max_length:
            padded = np.zeros(max_length)
            padded[:len(f)] = f
            standardized_features.append(padded)
        else:
            standardized_features.append(f)
    
    return np.array(standardized_features), np.array(valid_labels), valid_paths

# Model training and evaluation with enhanced visualization
def train_model(features, labels, image_paths, model_type='random_forest', output_dir='model_evaluations'):
    """
    Train a machine learning model with k-fold cross-validation and enhanced visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a list to store all fold results
    all_results = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    all_test_preds = []
    all_test_labels = []
    all_test_paths = []
    
    for train_index, test_index in kf.split(features):
        print(f"Training fold {fold}")
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        test_paths = [image_paths[i] for i in test_index]
        
        print(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'svr':
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        elif model_type == 'knn':
            model = KNeighborsRegressor(n_neighbors=5)
        elif model_type == 'kmeans':
            model = KMeans(n_clusters=5, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"Fold {fold} - Training RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.2f}")
        print(f"Fold {fold} - Testing RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.2f}")
        
        # Store results for this fold
        fold_results = {
            'fold': fold,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        all_results.append(fold_results)
        
        all_test_preds.extend(test_pred.tolist())
        all_test_labels.extend(y_test.tolist())
        all_test_paths.extend(test_paths)
        
        fold += 1
    
    # Create comprehensive fold comparison chart
    results_df = pd.DataFrame(all_results)
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    results_df[['train_rmse', 'test_rmse']].plot(kind='bar', ax=plt.gca())
    plt.title(f'RMSE by Fold - {model_type.capitalize()} Model')
    plt.ylabel('RMSE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(3, 1, 2)
    results_df[['train_mae', 'test_mae']].plot(kind='bar', ax=plt.gca())
    plt.title(f'MAE by Fold - {model_type.capitalize()} Model')
    plt.ylabel('MAE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(3, 1, 3)
    results_df[['train_r2', 'test_r2']].plot(kind='bar', ax=plt.gca())
    plt.title(f'R² by Fold - {model_type.capitalize()} Model')
    plt.ylabel('R²')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type}_fold_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Train final model on all data
    if model_type == 'random_forest':
        final_model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'svr':
        final_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    elif model_type == 'knn':
        final_model = KNeighborsRegressor(n_neighbors=5)
    elif model_type == 'kmeans':
        final_model = KMeans(n_clusters=5, random_state=42)
    
    final_model.fit(features, labels)
    
    # Create more informative prediction vs actual plots
    plt.figure(figsize=(12, 10))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(all_test_labels, all_test_preds, alpha=0.6, c=np.abs(np.array(all_test_preds) - np.array(all_test_labels)), cmap='YlOrRd')
    plt.colorbar(label='Absolute Error')
    
    # Add perfect prediction line
    min_val = min(min(all_test_labels), min(all_test_preds))
    max_val = max(max(all_test_labels), max(all_test_preds))
    plt.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', linewidth=2)
    
    plt.xlabel('Actual Stair Count')
    plt.ylabel('Predicted Stair Count')
    plt.title(f'Actual vs Prediction ({model_type.capitalize()})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Histogram of errors
    plt.subplot(2, 2, 2)
    errors = np.array(all_test_preds) - np.array(all_test_labels)
    plt.hist(errors, bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Residual plot
    plt.subplot(2, 2, 3)
    plt.scatter(all_test_labels, errors, alpha=0.6, c=np.abs(errors), cmap='YlOrRd')
    plt.axhline(y=0, color='blue', linestyle='--', linewidth=2)
    plt.xlabel('Actual Stair Count')
    plt.ylabel('Residual (Predicted - Actual)')
    plt.title('Residual Plot')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(label='Absolute Error')
    
    # Grouped error analysis
    plt.subplot(2, 2, 4)
    all_test_labels_array = np.array(all_test_labels)
    all_test_preds_array = np.array(all_test_preds)
    
    # Group by actual stair count
    unique_counts = sorted(set(all_test_labels))
    mean_errors = []
    std_errors = []
    
    for count in unique_counts:
        indices = np.where(all_test_labels_array == count)[0]
        count_errors = all_test_preds_array[indices] - all_test_labels_array[indices]
        mean_errors.append(np.mean(count_errors))
        std_errors.append(np.std(count_errors))
    
    plt.errorbar(unique_counts, mean_errors, yerr=std_errors, fmt='o', capsize=5, 
                 ecolor='black', markerfacecolor='orange', markeredgecolor='black')
    plt.axhline(y=0, color='blue', linestyle='--', linewidth=2)
    plt.xlabel('Actual Stair Count')
    plt.ylabel('Mean Error')
    plt.title('Mean Error by Stair Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type}_prediction_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create JSON results file for visualization
    results_json = {}
    
    for i, (image_path, actual, pred) in enumerate(zip(all_test_paths, all_test_labels, all_test_preds)):
        image_name = os.path.basename(image_path)
        
        if image_name not in results_json:
            results_json[image_name] = []
        
        results_json[image_name].append({
            'model': model_type,
            'preprocessing': 'standard',  # Assuming standard preprocessing
            'ground_truth': float(actual),
            'prediction': float(pred)
        })
    
    # Save the JSON file
    with open(f"{output_dir}/{model_type}_results.json", 'w') as f:
        json.dump(results_json, f, indent=4)
    
    return final_model, results_json

# Combined visualization function
def combine_model_results(output_dir='model_evaluations'):
    """
    Combine results from different models and create comparative visualizations.
    """
    # Find all model result files
    model_files = [f for f in os.listdir(output_dir) if f.endswith('_results.json')]
    
    if not model_files:
        print("No model result files found.")
        return
    
    # Load and combine all results
    combined_results = {}
    model_names = []
    
    for file in model_files:
        model_name = file.split('_')[0]
        model_names.append(model_name)
        
        with open(os.path.join(output_dir, file), 'r') as f:
            results = json.load(f)
            
            for image_name, evaluations in results.items():
                if image_name not in combined_results:
                    combined_results[image_name] = []
                
                combined_results[image_name].extend(evaluations)
    
    # Save combined results
    with open(f"{output_dir}/combined_results.json", 'w') as f:
        json.dump(combined_results, f, indent=4)
    
    # Create a DataFrame for easier analysis
    rows = []
    for image_name, evaluations in combined_results.items():
        for evaluation in evaluations:
            row = {
                'image_name': image_name,
                **evaluation,
                'error': abs(evaluation['prediction'] - evaluation['ground_truth']),
                'relative_error': abs(evaluation['prediction'] - evaluation['ground_truth']) / 
                                 max(evaluation['ground_truth'], 1e-10) * 100,
                'squared_error': (evaluation['prediction'] - evaluation['ground_truth'])**2
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Create comprehensive model comparison visualization
    plt.figure(figsize=(16, 14))
    
    # Model comparison by error metrics
    plt.subplot(3, 2, 1)
    model_stats = df.groupby('model').agg({
        'error': ['mean', 'median', 'std'],
        'relative_error': ['mean', 'median'],
        'squared_error': ['mean']
    })
    model_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in model_stats.columns.values]
    model_stats['rmse'] = np.sqrt(model_stats['squared_error_mean'])
    
    metrics = ['error_mean', 'error_median', 'rmse']
    model_stats[metrics].plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('Model Comparison - Error Metrics')
    plt.ylabel('Error')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Model comparison by relative error
    plt.subplot(3, 2, 2)
    metrics = ['relative_error_mean', 'relative_error_median']
    model_stats[metrics].plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('Model Comparison - Relative Error (%)')
    plt.ylabel('Relative Error (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Error distribution
    plt.subplot(3, 2, 3)
    sns.boxplot(x='model', y='error', data=df)
    plt.title('Error Distribution by Model')
    plt.ylabel('Absolute Error')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Relative error distribution
    plt.subplot(3, 2, 4)
    sns.boxplot(x='model', y='relative_error', data=df)
    plt.title('Relative Error Distribution by Model (%)')
    plt.ylabel('Relative Error (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Error heatmap
    plt.subplot(3, 2, 5)
    pivot_table = df.pivot_table(values='error', index='image_name', columns='model', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap=custom_cmap)
    plt.title('Mean Error by Image and Model')
    plt.xticks(rotation=45)
    
    # Best model count
    plt.subplot(3, 2, 6)
    best_models = df.loc[df.groupby('image_name')['error'].idxmin()]['model']
    best_model_counts = best_models.value_counts()
    best_model_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, 
                          colors=plt.cm.Paired(np.linspace(0, 1, len(best_model_counts))))
    plt.title('Best Model by Count')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed error analysis
    plt.figure(figsize=(16, 14))
    
    # Error distribution histogram
    plt.subplot(2, 2, 1)
    for model in model_names:
        model_df = df[df['model'] == model]
        sns.kdeplot(model_df['error'], label=model, fill=True, alpha=0.3)
    plt.title('Error Distribution Density by Model')
    plt.xlabel('Absolute Error')
    plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Actual vs predicted for best model
    plt.subplot(2, 2, 2)
    best_model = model_stats.loc[model_stats['error_mean'].idxmin()].name
    best_model_df = df[df['model'] == best_model]
    
    plt.scatter(best_model_df['ground_truth'], best_model_df['prediction'], 
               alpha=0.7, c=best_model_df['error'], cmap='YlOrRd')
    
    min_val = min(min(best_model_df['ground_truth']), min(best_model_df['prediction']))
    max_val = max(max(best_model_df['ground_truth']), max(best_model_df['prediction']))
    plt.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', linewidth=2)
    
    plt.xlabel('Actual Stair Count')
    plt.ylabel('Predicted Stair Count')
    plt.title(f'Best Model: {best_model.capitalize()} - Actual vs Predicted')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(label='Absolute Error')
    
    # Error by stair count
    plt.subplot(2, 2, 3)
    error_by_count = df.groupby(['ground_truth', 'model'])['error'].mean().reset_index()
    
    for model in model_names:
        model_data = error_by_count[error_by_count['model'] == model]
        plt.plot(model_data['ground_truth'], model_data['error'], 'o-', label=model)
    
    plt.xlabel('Actual Stair Count')
    plt.ylabel('Mean Absolute Error')
    plt.title('Error by Stair Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Top 5 hardest images
    plt.subplot(2, 2, 4)
    hardest_images = df.groupby('image_name')['error'].mean().sort_values(ascending=False).head(5)
    hardest_images.plot(kind='bar', ax=plt.gca())
    plt.title('Top 5 Hardest Images to Predict')
    plt.ylabel('Mean Absolute Error')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot correlation between features and errors
    return df

# Enhanced visualization functions
def visualize_image_evaluations(df, output_dir='evaluation_visualizations'):
    """Create visualizations for each image with enhanced plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    unique_images = df['image_name'].unique()
    
    for image_name in unique_images:
        image_df = df[df['image_name'] == image_name]
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig)
        
        # Prediction vs Ground Truth
        ax1 = fig.add_subplot(gs[0, :2])
        pivot_df = image_df.pivot(index='preprocessing', columns='model', values='prediction')
        pivot_df.plot(kind='bar', ax=ax1)
        ax1.axhline(y=image_df['ground_truth'].iloc[0], color='r', linestyle='-', linewidth=2)
        ax1.text(0, image_df['ground_truth'].iloc[0]*1.05, f"Ground Truth: {image_df['ground_truth'].iloc[0]}", 
                color='r', fontweight='bold')
        ax1.set_title(f'Predictions vs Ground Truth for {image_name}')
        ax1.set_ylabel('Stair Count')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Absolute Error
        ax2 = fig.add_subplot(gs[0, 2:])
        sns.heatmap(image_df.pivot(index='preprocessing', columns='model', values='error'),
                   annot=True, cmap=custom_cmap, ax=ax2)
        ax2.set_title('Absolute Error')
        
        # Relative Error
        ax3 = fig.add_subplot(gs[1, :2])
        sns.heatmap(image_df.pivot(index='preprocessing', columns='model', values='relative_error'),
                   annot=True, fmt='.1f', cmap=custom_cmap, ax=ax3)
        ax3.set_title('Relative Error (%)')
        
        # Error across models
        ax4 = fig.add_subplot(gs[1, 2:])
        image_df.plot(kind='bar', x='model', y='error', ax=ax4, legend=False)
        ax4.set_title('Error Comparison Across Models')
        ax4.set_ylabel('Absolute Error')
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        ax4.tick_params(axis='x', rotation=45)
        
        # Best combinations
        ax5 = fig.add_subplot(gs[2, :])
        best_df = image_df.sort_values('error')
        n_best = min(5, len(best_df))
        best_combinations = best_df.iloc[:n_best][['preprocessing', 'model', 'prediction', 'ground_truth', 'error', 'relative_error']]
        
        cell_text = []
        for _, row in best_combinations.iterrows():
            cell_text.append([
                row['preprocessing'], 
                row['model'], 
                f"{row['prediction']:.2f}", 
                f"{row['ground_truth']:.2f}", 
                f"{row['error']:.2f}",
                f"{row['relative_error']:.1f}%"
            ])
        
        ax5.axis('off')
        table = ax5.table(
            cellText=cell_text,
            colLabels=['Preprocessing', 'Model', 'Pred', 'Truth', 'Error', 'Rel. Error'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax5.set_title('Best Model-Preprocessing Combinations')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{image_name.split('.')[0]}_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_visualization(df, output_dir='evaluation_visualizations'):
    """Create enhanced summary visualizations for all images."""
    os.makedirs(output_dir, exist_ok=True)
    
    model_preprocessing_stats = df.groupby(['model', 'preprocessing']).agg({
        'error': ['mean', 'median', 'std'],
        'relative_error': ['mean', 'median', 'std'],
        'squared_error': ['mean']
    }).reset_index()
    
    model_preprocessing_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in model_preprocessing_stats.columns.values]
    model_preprocessing_stats['rmse'] = np.sqrt(model_preprocessing_stats['squared_error_mean'])
    
    # Create a more comprehensive multi-page visualization
    # Page 1: Overall performance
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(3, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    pivoted = df.pivot_table(values='error', index='preprocessing', columns='model', aggfunc='mean')
    sns.heatmap(pivoted, annot=True, cmap=custom_cmap, ax=ax1)
    ax1.set_title('Mean Absolute Error by Model and Preprocessing')
    
    ax2 = fig.add_subplot(gs[0, 1])
    pivoted = df.pivot_table(values='relative_error', index='preprocessing', columns='model', aggfunc='mean')
    sns.heatmap(pivoted, annot=True, fmt='.1f', cmap=custom_cmap, ax=ax2)
    ax2.set_title('Mean Relative Error (%) by Model and Preprocessing')
    
    ax3 = fig.add_subplot(gs[1, 0])
    pivoted = df.pivot_table(values='squared_error', index='preprocessing', columns='model', aggfunc='mean')
    pivoted = pivoted.apply(np.sqrt)  # Convert to RMSE
    sns.heatmap(pivoted, annot=True, fmt='.2f', cmap=custom_cmap, ax=ax3)
    ax3.set_title('RMSE by Model and Preprocessing')
    
    ax4 = fig.add_subplot(gs[1, 1])
    # Calculate success rate (predictions within 1 step of ground truth)
    df['success'] = (df['error'] <= 1).astype(int)
    pivoted = df.pivot_table(values='success', index='preprocessing', columns='model', aggfunc='mean')
    sns.heatmap(pivoted, annot=True, fmt='.2%', cmap='YlGn', ax=ax4)
    ax4.set_title('Success Rate (Error ≤ 1) by Model and Preprocessing')
    
    ax5 = fig.add_subplot(gs[2, :])
    best_combinations = model_preprocessing_stats.sort_values('error_mean')
    
    top_15 = best_combinations.head(15)
    
    x = range(len(top_15))
    width = 0.25
    
    ax5.bar([i - width for i in x], top_15['error_mean'], width, label='Mean Error', color='skyblue')
    ax5.bar([i for i in x], top_15['error_median'], width, label='Median Error', color='lightgreen')
    ax5.bar([i + width for i in x], top_15['rmse'], width, label='RMSE', color='salmon')
    
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"{row['model']} + {row['preprocessing']}" for _, row in top_15.iterrows()], rotation=45, ha='right')
    ax5.set_ylabel('Error')
    ax5.set_title('Top 15 Model-Preprocessing Combinations by Error')
    ax5.legend()
    ax5.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Page 2: Error distribution and analysis
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(3, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(x='model', y='error', data=df, ax=ax1)
    ax1.set_title('Error Distribution by Model')
    ax1.set_ylabel('Absolute Error')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(x='preprocessing', y='error', data=df, ax=ax2)
    ax2.set_title('Error Distribution by Preprocessing')
    ax2.set_ylabel('Absolute Error')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='x', rotation=45)
    
    ax3 = fig.add_subplot(gs[1, 0])
    sns.violinplot(x='model', y='relative_error', data=df, ax=ax3)
    ax3.set_title('Relative Error Distribution by Model')
    ax3.set_ylabel('Relative Error (%)')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    ax3.tick_params(axis='x', rotation=45)
    
    ax4 = fig.add_subplot(gs[1, 1])
    sns.violinplot(x='preprocessing', y='relative_error', data=df, ax=ax4)
    ax4.set_title('Relative Error Distribution by Preprocessing')
    ax4.set_ylabel('Relative Error (%)')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    ax4.tick_params(axis='x', rotation=45)
    
    ax5 = fig.add_subplot(gs[2, :])
    error_by_count = df.groupby(['ground_truth', 'model'])['error'].mean().reset_index()
    
    for model in df['model'].unique():
        model_data = error_by_count[error_by_count['model'] == model]
        ax5.plot(model_data['ground_truth'], model_data['error'], 'o-', label=model)
    
    ax5.set_xlabel('Actual Stair Count')
    ax5.set_ylabel('Mean Absolute Error')
    ax5.set_title('Error by Stair Count')
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Page 3: Best and worst performing images
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(2, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    hardest_images = df.groupby('image_name')['error'].mean().sort_values(ascending=False).head(5)
    hardest_images.plot(kind='bar', ax=ax1)
    ax1.set_title('Top 5 Hardest Images to Predict')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = fig.add_subplot(gs[0, 1])
    easiest_images = df.groupby('image_name')['error'].mean().sort_values().head(5)
    easiest_images.plot(kind='bar', ax=ax2)
    ax2.set_title('Top 5 Easiest Images to Predict')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='x', rotation=45)
    
    ax3 = fig.add_subplot(gs[1, :])
    best_models = df.loc[df.groupby('image_name')['error'].idxmin()]['model']
    best_model_counts = best_models.value_counts()
    best_model_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, 
                          colors=plt.cm.Paired(np.linspace(0, 1, len(best_model_counts))), ax=ax3)
    ax3.set_title('Best Model by Count')
    ax3.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/image_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

# Main function to run the entire pipeline
def main():
    # Load data from the database dump
    image_paths = []
    labels = []
    
    with open('data/stairsData_dump', 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith("COPY public.images_data"):
            break
    
    for line in lines[lines.index(line) + 1:]:
        if line.strip() == '\\.':
            break
        parts = line.strip().split('\t')
        if len(parts) == 4:
            image_path = parts[1]
            label = int(parts[2])
            
            if image_path.startswith('.'):
                image_path = image_path[2:]
            
            if os.path.isfile(image_path):
                image_paths.append(image_path)
                labels.append(label)
            else:
                print(f"File not found: {image_path}")
    
    print(f"Collected {len(image_paths)} image paths and {len(labels)} labels")
    
    features, labels, valid_paths = prepare_dataset(image_paths, labels)
    
    if features.size == 0 or labels.size == 0:
        print("No valid data to train the model.")
        return
    
    # Train and evaluate models
    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'knn': KNeighborsRegressor(n_neighbors=5),
        'kmeans': KMeans(n_clusters=5, random_state=42)
    }
    
    all_results = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name} model...")
        trained_model, results_json = train_model(features, labels, valid_paths, model_type=model_name, output_dir='results/visualisation/machine_learning')
        joblib.dump(trained_model, f"src/models/{model_name}_model.pkl")
        print(f"{model_name} model training complete and saved to src/models/{model_name}_model.pkl")
        
        all_results[model_name] = results_json
    
    # Combine results and create visualizations
    combine_model_results(output_dir='results/visualisation/machine_learning')
    
    # Load combined results for visualization
    with open('results/visualisation/machine_learning/combined_results.json', 'r') as f:
        combined_results = json.load(f)
    
    # Convert combined_results into a DataFrame for further processing
    rows = []
    for image_name, evaluations in combined_results.items():
        for evaluation in evaluations:
            row = {
                'image_name': image_name,
                **evaluation,
                'error': abs(evaluation['prediction'] - evaluation['ground_truth']),
                'relative_error': abs(evaluation['prediction'] - evaluation['ground_truth']) / 
                                 max(evaluation['ground_truth'], 1e-10) * 100,
                'squared_error': (evaluation['prediction'] - evaluation['ground_truth'])**2
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    
    # Create enhanced visualizations
    visualize_image_evaluations(df, output_dir='results/visualisation/machine_learning')
    create_summary_visualization(df, output_dir='results/visualisation/machine_learning')

if __name__ == "__main__":
    main()
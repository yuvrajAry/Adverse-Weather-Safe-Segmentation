#!/usr/bin/env python3
"""
Generate visualizations for IDDAW project report
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import pandas as pd

# Set the style for plots
plt.style.use('ggplot')

def extract_metrics_from_files(results_dir):
    """Extract metrics from result files"""
    metrics = {}
    
    # Pattern to match metrics in the result files
    pattern = r'mIoU=(\d+\.\d+), safe_mIoU=(\d+\.\d+)'
    
    # Find all result files
    result_files = glob.glob(os.path.join(results_dir, '*_results.txt'))
    
    for file_path in result_files:
        model_name = os.path.basename(file_path).replace('_results.txt', '')
        
        with open(file_path, 'r') as f:
            content = f.read()
            match = re.search(pattern, content)
            
            if match:
                miou = float(match.group(1))
                safe_miou = float(match.group(2))
                metrics[model_name] = {
                    'mIoU': miou,
                    'safe_mIoU': safe_miou
                }
    
    return metrics

def create_bar_chart(metrics, output_dir):
    """Create bar chart comparing model performance"""
    if not metrics:
        print("No metrics found to create bar chart")
        return
    
    # Prepare data for plotting
    models = list(metrics.keys())
    miou_values = [metrics[model]['mIoU'] for model in models]
    safe_miou_values = [metrics[model]['safe_mIoU'] for model in models]
    
    # Improve model names for display
    display_names = [name.replace('_', ' ').upper() for name in models]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set width of bars
    bar_width = 0.35
    
    # Set position of bars on x axis
    r1 = np.arange(len(models))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    ax.bar(r1, miou_values, width=bar_width, label='mIoU', color='#3498db')
    ax.bar(r2, safe_miou_values, width=bar_width, label='safe_mIoU', color='#2ecc71')
    
    # Add labels and title
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks([r + bar_width/2 for r in range(len(models))])
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    
    # Add values on top of bars
    for i, v in enumerate(miou_values):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    for i, v in enumerate(safe_miou_values):
        ax.text(i + bar_width, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'model_performance_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"Bar chart saved to {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)

def create_radar_chart(metrics, output_dir):
    """Create radar chart for model comparison"""
    if not metrics:
        print("No metrics found to create radar chart")
        return
    
    # Prepare data for plotting
    models = list(metrics.keys())
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(models)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], [name.replace('_', ' ').upper() for name in models], size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.2, 0.3, 0.4], ["0.1", "0.2", "0.3", "0.4"], color="grey", size=10)
    plt.ylim(0, 0.5)
    
    # Plot mIoU values
    miou_values = [metrics[model]['mIoU'] for model in models]
    miou_values += miou_values[:1]  # Close the loop
    ax.plot(angles, miou_values, linewidth=2, linestyle='solid', label='mIoU')
    ax.fill(angles, miou_values, alpha=0.25)
    
    # Plot safe_mIoU values
    safe_miou_values = [metrics[model]['safe_mIoU'] for model in models]
    safe_miou_values += safe_miou_values[:1]  # Close the loop
    ax.plot(angles, safe_miou_values, linewidth=2, linestyle='solid', label='safe_mIoU')
    ax.fill(angles, safe_miou_values, alpha=0.25)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title('Model Performance Radar Chart', size=15, fontweight='bold', y=1.1)
    
    # Save figure
    output_path = os.path.join(output_dir, 'model_performance_radar.png')
    plt.savefig(output_path, dpi=300)
    print(f"Radar chart saved to {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)

def create_image_grid(output_dir):
    """Create image grid of sample results"""
    # Define model directories
    model_dirs = [
        'rgb_mbv3',
        'nir_fastscnn',
        'early4_mbv3',
        'mid_mbv3'
    ]
    
    # Find overlay images for each model
    overlay_images = {}
    for model in model_dirs:
        model_dir = os.path.join(output_dir, model)
        if os.path.exists(model_dir):
            # Find overlay images
            images = glob.glob(os.path.join(model_dir, '*_overlay.png'))
            if images:
                # Sort images to ensure consistent ordering
                images.sort()
                # Take first 3 images
                overlay_images[model] = images[:3]
    
    if not overlay_images:
        print("No overlay images found to create image grid")
        return
    
    # Create figure for the grid
    num_models = len(overlay_images)
    num_samples = min(3, min(len(images) for images in overlay_images.values()))
    
    fig, axes = plt.subplots(num_models, num_samples, figsize=(15, 4 * num_models))
    
    # Add images to the grid
    for i, (model, images) in enumerate(overlay_images.items()):
        for j in range(num_samples):
            if j < len(images):
                img = Image.open(images[j])
                if num_models > 1:
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(f"{model.replace('_', ' ').upper()}\nSample {j+1}")
                    axes[i, j].axis('off')
                else:
                    axes[j].imshow(img)
                    axes[j].set_title(f"{model.replace('_', ' ').upper()}\nSample {j+1}")
                    axes[j].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'sample_results_grid.png')
    plt.savefig(output_path, dpi=300)
    print(f"Image grid saved to {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)

def create_html_report(metrics, output_dir):
    """Create an HTML report with all visualizations"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IDDAW Project Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .chart {
                margin: 30px 0;
                text-align: center;
            }
            .chart img {
                max-width: 100%;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                border-radius: 4px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px 15px;
                border: 1px solid #ddd;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .footer {
                margin-top: 50px;
                text-align: center;
                font-size: 0.8em;
                color: #7f8c8d;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>IDDAW Project Results</h1>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>mIoU</th>
                    <th>Safe mIoU</th>
                </tr>
    """
    
    # Add metrics to the table
    for model, values in metrics.items():
        display_name = model.replace('_', ' ').upper()
        html_content += f"""
                <tr>
                    <td>{display_name}</td>
                    <td>{values['mIoU']:.3f}</td>
                    <td>{values['safe_mIoU']:.3f}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>Performance Comparison</h2>
            <div class="chart">
                <h3>Bar Chart Comparison</h3>
                <img src="model_performance_comparison.png" alt="Model Performance Comparison">
            </div>
            
            <div class="chart">
                <h3>Radar Chart Comparison</h3>
                <img src="model_performance_radar.png" alt="Model Performance Radar Chart">
            </div>
            
            <h2>Sample Results</h2>
            <div class="chart">
                <h3>Sample Segmentation Results</h3>
                <img src="sample_results_grid.png" alt="Sample Segmentation Results">
            </div>
            
            <div class="footer">
                <p>Generated for IDDAW Project Report</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    output_path = os.path.join(output_dir, 'results_report.html')
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {output_path}")

def main():
    """Main function to generate all visualizations"""
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'project', 'outputs', 'report')
    
    print(f"Generating visualizations from results in: {results_dir}")
    
    # Create visualizations directory if it doesn't exist
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    # Extract metrics from result files
    metrics = extract_metrics_from_files(results_dir)
    
    if not metrics:
        print("No metrics found in result files")
        return
    
    print(f"Found metrics for {len(metrics)} models: {', '.join(metrics.keys())}")
    
    # Create visualizations
    create_bar_chart(metrics, results_dir)
    create_radar_chart(metrics, results_dir)
    create_image_grid(results_dir)
    create_html_report(metrics, results_dir)
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    main()
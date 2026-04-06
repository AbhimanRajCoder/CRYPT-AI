import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(BASE_DIR, "runs", "test", "metrics", "results.json")
CLASS_METRICS_CSV = os.path.join(BASE_DIR, "runs", "test", "metrics", "class_metrics.csv")
OUTPUT_IMG = os.path.join(BASE_DIR, "runs", "test", "graphs", "14_Metrics_Summary_Table.png")

def main():
    # Load data
    with open(RESULTS_JSON, 'r') as f:
        global_results = json.load(f)
        
    df = pd.read_csv(CLASS_METRICS_CSV)
    
    # Format global metrics into a dataframe row
    global_row = pd.DataFrame([{
        "Class": "OVERALL (Mean)",
        "IoU": f"{global_results['mean_IoU']*100:.1f}%",
        "Precision": f"{global_results['mean_Precision']*100:.1f}%",
        "Recall": f"{global_results['mean_Recall']*100:.1f}%",
        "F1-Score": f"{global_results['mean_F1']*100:.1f}%"
    }])
    
    # Optional mAP row
    map_row = pd.DataFrame([{
        "Class": "mAP@50 (Global)",
        "IoU": "-",
        "Precision": "-",
        "Recall": "-",
        "F1-Score": f"{global_results['mAP@50 (Global AP)']*100:.1f}%"
    }])

    # Format class metrics
    for col in ["IoU", "Precision", "Recall", "F1-Score"]:
        df[col] = df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")

    # Combine into final table dataframe
    final_df = pd.concat([df, global_row, map_row], ignore_index=True)
    
    # Draw via matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    table = ax.table(
        cellText=final_df.values,
        colLabels=final_df.columns,
        loc='center',
        cellLoc='center'
    )
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1a237e')  # Deep blue header
        elif row >= len(final_df) - 1:  # The summary rows at bottom
            cell.set_facecolor('#e8eaf6')
            cell.set_text_props(weight='bold')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f5f5f5')
                
    # Add title
    plt.title("🏜️ Off-Road Segmentation Metrics Summary", fontweight='bold', fontsize=18, y=0.95)
    
    # Save
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"✅ Saved metrics table visual to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()

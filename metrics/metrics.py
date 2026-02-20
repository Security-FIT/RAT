#/usr/bin/env python3
import re
import pandas as pd
import glob
import matplotlib.pyplot as plt

def parse_metrics(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    
    # Extraction Logic
    data = {
        "File": file_path.split('/')[-1],
    }

    # Gradient Norms
    ref_grad_match = re.search(r'Reference gradient norms - Mean: ([\d.e+-]+), Std: ([\d.e+-]+)', text)
    if ref_grad_match:
        data["Ref_Grad_Mean"] = float(ref_grad_match.group(1))
        data["Ref_Grad_Std"] = float(ref_grad_match.group(2))
    
    test_grad_match = re.search(r'Test gradient norms - Mean: ([\d.e+-]+), Std: ([\d.e+-]+)', text)
    if test_grad_match:
        data["Test_Grad_Mean"] = float(test_grad_match.group(1))
        data["Test_Grad_Std"] = float(test_grad_match.group(2))

    grad_ratio_match = re.search(r'Average gradient norm ratio \(reference/test\): ([\d.e+-]+)', text)
    if grad_ratio_match:
        data["Grad_Ratio"] = float(grad_ratio_match.group(1))

    # Delta Trace Table
    # Parse table rows like: "attn_map        | 7.802e-01       | 1.205e+00"
    table_keys = ["attn_map", "mlp_out", "combined", "residual", "final_out", "emb", "logits"]
    
    # Store noise and zero metrics separately first to group them
    noise_metrics = {}
    zero_metrics = {}
    
    for key in table_keys:
        # Regex to find the row for this key. 
        # Expects: key | float | float
        # We handle variable whitespace
        match = re.search(rf'{key}\s+\|\s+([\d.e+-]+)\s+\|\s+([\d.e+-]+)', text)
        if match:
            noise_metrics[f"Act_Delta_{key}_Noise"] = float(match.group(1))
            zero_metrics[f"Act_Delta_{key}_Zero"] = float(match.group(2))

    # Add to data in grouped order
    data.update(noise_metrics)
    data.update(zero_metrics)

    # Margins
    # Δmargin (NOISE): 1.428e+00  (normalized: 1.338e-01)
    margin_noise_match = re.search(r'Δmargin \(NOISE\)\s*:\s*([\d.e+-]+)\s*\(normalized:\s*([\d.e+-]+)\)', text)
    if margin_noise_match:
        data["Delta_Margin_Noise"] = float(margin_noise_match.group(1))
        data["Delta_Margin_Noise_Norm"] = float(margin_noise_match.group(2))

    # Δmargin (ZERO) : 4.963e+00  (normalized: 4.649e-01)
    margin_zero_match = re.search(r'Δmargin \(ZERO\)\s*:\s*([\d.e+-]+)\s*\(normalized:\s*([\d.e+-]+)\)', text)
    if margin_zero_match:
        data["Delta_Margin_Zero"] = float(margin_zero_match.group(1))
        data["Delta_Margin_Zero_Norm"] = float(margin_zero_match.group(2))

    # Ratios
    # R_attn/mlp (RMS): mean=2.168e-02, std=3.106e-03
    ratio_match = re.search(r'R_attn/mlp \(RMS\): mean=([\d.e+-]+), std=([\d.e+-]+)', text)
    if ratio_match:
        data["Attn_MLP_Ratio_Mean"] = float(ratio_match.group(1))
        data["Attn_MLP_Ratio_Std"] = float(ratio_match.group(2))

    return data

# 1. Collect all output files
log_files = glob.glob("metrics_FFAttn2_*.out")
log_files += glob.glob("metrics_finetune_FFAttn2_*.out")

# 2. Parse all files into a list of dictionaries
results = []
for file in log_files:
    try:
        results.append(parse_metrics(file))
    except Exception as e:
        print(f"Skipping {file} due to error: {e}")

# 3. Create DataFrame and Sort
df = pd.DataFrame(results)
df = df.sort_values("File")

df.to_csv("metrics_summary.csv", index=False)
print("Metrics summary saved to metrics_summary.csv")

#####################################################

def extract_epoch(filename: str):
    return (int(filename[-13:-12]) + 5) if 'finetune' in filename else int(filename[-13:-12])

df['Epoch'] = df['File'].apply(extract_epoch)
df_sorted_epoch = df.sort_values('Epoch')

# Set global font size
plt.rcParams.update({'font.size': 12})

# Colors (custom, not mpl defaults)
COLOR_MARGIN = '#d35400'    # Pumpkin Orange
COLOR_ATTN   = '#2980b9'    # Belize Hole Blue
COLOR_FINAL  = '#27ae60'    # Nephritis Green
COLOR_EMB    = '#8e44ad'    # Wisteria Purple

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

epochs = df_sorted_epoch['Epoch']
x_lims = (epochs.min(), epochs.max())

# --- Left Panel: Margins and Attn/MLP Ratio ---
# Margins on Left Axis (Orange) - AREA CHART
if 'Delta_Margin_Zero_Norm' in df_sorted_epoch.columns:
    ax1.fill_between(epochs, 0, df_sorted_epoch['Delta_Margin_Zero_Norm'],
                     color=COLOR_MARGIN, alpha=0.4)
    ax1.plot(epochs, df_sorted_epoch['Delta_Margin_Zero_Norm'], 
             marker='o', linestyle='-', color=COLOR_MARGIN, alpha=0.9, label=r'$\Delta m_n$ (Zero Ref)')

if 'Delta_Margin_Noise_Norm' in df_sorted_epoch.columns:
    ax1.fill_between(epochs, 0, df_sorted_epoch['Delta_Margin_Noise_Norm'],
                     color=COLOR_MARGIN, alpha=0.15, hatch='///')
    ax1.plot(epochs, df_sorted_epoch['Delta_Margin_Noise_Norm'], 
             marker='x', linestyle='--', linewidth=1.5, color=COLOR_MARGIN, label=r'$\Delta m_n$ (Noise Ref)')

# Attn/MLP Ratio on Right Axis (Blue)
ax1_twin = ax1.twinx()
# Ensure twin axis is on top for visibility
ax1.set_zorder(1)  # default
ax1_twin.set_zorder(2) 
ax1_twin.patch.set_visible(False) # Prevent hiding ax1

if 'Attn_MLP_Ratio_Mean' in df_sorted_epoch.columns:
    ax1_twin.plot(epochs, df_sorted_epoch['Attn_MLP_Ratio_Mean'],
             marker='^', linestyle='-', color=COLOR_ATTN, linewidth=1.5, label=r'Attn/MLP ratio ($r_b$)')

# ax1.axvline(x=5.5, color='gray', linestyle=':', label='Finetuning Start', alpha=0.8)
ax1.set_xlim(x_lims)
ax1.set_ylim(bottom=0)
# ax1.set_title('Normalized Delta Margin & Ratio')
ax1.set_xlabel('Epoch')
ax1.set_xticks(epochs)

ax1.set_ylabel(r'Normalized Delta Margin ($\Delta m_n$)', color=COLOR_MARGIN)
ax1_twin.set_ylabel(r'Branch ratio ($r_b$)', color=COLOR_ATTN)

ax1.tick_params(axis='y', labelcolor=COLOR_MARGIN)
ax1_twin.tick_params(axis='y', labelcolor=COLOR_ATTN)
ax1.grid(True, alpha=0.3)

# Combine legends for Left Panel
lines1, labels1 = ax1.get_legend_handles_labels()
lines1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()

# Combine lists
all_lines = lines1 + lines1_twin
all_labels = labels1 + labels1_twin

# Reorder: Move 'Finetuning Start' to the end
# try:
#     ft_idx = all_labels.index('Finetuning Start')
#     ft_line = all_lines.pop(ft_idx)
#     ft_label = all_labels.pop(ft_idx)
#     all_lines.append(ft_line)
#     all_labels.append(ft_label)
# except ValueError:
#     pass # Label not found

ax1.legend(all_lines, all_labels, loc='upper right')


# --- Right Panel: Activations (Final Out & Emb) ---
# Stacked Area Chart
# Base: Emb (Purple)
# Top: Final Out (Green) stacked on Emb

# Helper to plot stacked areas
def plot_stacked(ax, epochs, data_emb, data_final, label_suffix, line_style, alpha_fill, hatch=None):
    # Calculate accumulated values for stacking
    y_emb = data_emb
    y_total = data_emb + data_final
    
    # Fill Emb Area (Bottom)
    ax.fill_between(epochs, 0, y_emb, color=COLOR_EMB, alpha=alpha_fill, 
                    hatch=hatch, label=f'Emb ({label_suffix})')
    
    # Fill Final Out Area (Top)
    ax.fill_between(epochs, y_emb, y_total, color=COLOR_FINAL, alpha=alpha_fill, 
                    hatch=hatch, label=f'Final Out ({label_suffix})')
    
    # Plot Border Lines
    ax.plot(epochs, y_emb, color=COLOR_EMB, linestyle=line_style, linewidth=1, alpha=0.8)
    ax.plot(epochs, y_total, color=COLOR_FINAL, linestyle=line_style, linewidth=1.5, alpha=1.0) # Top of stack

# Plot Zero Condition (Solid, stronger fill)
if 'Act_Delta_emb_Zero' in df_sorted_epoch.columns and 'Act_Delta_final_out_Zero' in df_sorted_epoch.columns:
    plot_stacked(ax2, epochs, 
                 df_sorted_epoch['Act_Delta_emb_Zero'], 
                 df_sorted_epoch['Act_Delta_final_out_Zero'], 
                 'Zero', '-', 0.4)

# Plot Noise Condition (Dashed, lighter fill/hatched)
if 'Act_Delta_emb_Noise' in df_sorted_epoch.columns and 'Act_Delta_final_out_Noise' in df_sorted_epoch.columns:
    plot_stacked(ax2, epochs, 
                 df_sorted_epoch['Act_Delta_emb_Noise'], 
                 df_sorted_epoch['Act_Delta_final_out_Noise'], 
                 'Noise', '--', 0.15, hatch='///')

# ax2.axvline(x=5.5, color='gray', linestyle=':', alpha=0.8)
ax2.set_xlim(x_lims)
ax2.set_ylim(bottom=0)  # Start Y-axis at 0 for better area visualization
# ax2.set_title('Activation Deltas (Stacked)')
ax2.set_xlabel('Epoch')
ax2.set_xticks(epochs)

# Combined Y-label
ax2.set_ylabel(r'Normalized Activation Delta ($\Delta h_n$)', color='black')

ax2.grid(True, alpha=0.3)

# Combine legends for Right Panel
# Manually create handles for clearer legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color=COLOR_FINAL, linestyle='-', linewidth=2, label=r'$\Delta h_n(\mathbf{Z}_{\text{RIB}})$ (Zero Ref)'),
    Line2D([0], [0], color=COLOR_EMB, linestyle='-', linewidth=2, label=r'$\Delta h_n(\mathbf{e})$ (Zero Ref)'),
    Line2D([0], [0], color=COLOR_FINAL, linestyle='--', linewidth=2, label=r'$\Delta h_n(\mathbf{Z}_{\text{RIB}})$ (Noise Ref)'),
    Line2D([0], [0], color=COLOR_EMB, linestyle='--', linewidth=2, label=r'$\Delta h_n(\mathbf{e})$ (Noise Ref)'),
    # Line2D([0], [0], color='gray', linestyle=':', label='Finetuning Start'),
]
ax2.legend(handles=legend_elements, loc='upper right')

fig.tight_layout()
fig.savefig('delta_margin_plot.pdf')
# plt.show()

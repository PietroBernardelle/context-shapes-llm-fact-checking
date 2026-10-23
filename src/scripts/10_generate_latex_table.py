
from argparse import ArgumentParser
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import re
from prompt_utils import project_root, load_prompt_config
from evaluation_utils import read_subset_keys, filter_subset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support

DATASETS = ["Hover", "FEVEROUS", "ClimateFever"] 

def normalize_dataset_name(name):
    # Map common variations
    n = name.lower()
    if 'climate' in n: return 'ClimateFever'
    if 'feverous' in n: return 'FEVEROUS'
    if 'hover' in n: return 'Hover'
    return name

def get_metrics_for_df(df):
    """
    Calculates Acc, F1, Precision (SUP/REF), Recall (SUP/REF)
    """
    if 'label' not in df.columns or 'pred_label' not in df.columns:
        return {}
    
    df = df[df["label"].isin([0, 1]) & df["pred_label"].isin([0, 1])]
    if len(df) == 0:
        return {}
        
    y_true = df['label']
    y_pred = df['pred_label']
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=0)
    
    # The prompt and preprocessing use 0=SUPPORTS, 1=REFUTES.
    pos_label = 0
    neg_label = 1

    prec_sup = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    prec_ref = precision_score(y_true, y_pred, pos_label=neg_label, zero_division=0)
    
    rec_sup = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    rec_ref = recall_score(y_true, y_pred, pos_label=neg_label, zero_division=0)
    
    return {
        "Acc": acc,
        "F1": f1,
        "Prec_SUP": prec_sup,
        "Prec_REF": prec_ref,
        "Rec_SUP": rec_sup,
        "Rec_REF": rec_ref
    }

def parse_filler_filename(filename):
    stem = Path(filename).stem
    length = None
    pos = None
    
    # res_multi_len4096_pos0p1_subset_seed42
    
    len_match = re.search(r"_len(\d+)", stem)
    if len_match:
        length = int(len_match.group(1))
        # Exclude 32k results (approx > 28000)
        if length > 28000:
            return None, None
            
    # Extract position
    # Patterns: pos0 (0%), pos1 (100%), pos0p1 (10%), pos0p05 (5%)
    pos_match = re.search(r"_pos([0-9p]+)", stem)
    if pos_match:
        pos_str = pos_match.group(1)
        if 'p' in pos_str:
            pos_val = float(pos_str.replace('p', '.'))
        else:
            pos_val = float(pos_str)
        pos = int(round(pos_val * 100))
        
    return length, pos

def get_depth_stats(model_name: str, dataset_name: str, results_dir: Path,
                    subset_keys: set[tuple[str, str]], allowed_lengths: list[int]) -> dict:
    # Scans for filler files
    model_dir = os.path.join(results_dir, model_name)
    if not os.path.exists(model_dir):
        return {}
    
    files = glob.glob(os.path.join(model_dir, "res_multi_len*_pos*.pqt"))
    
    data_points = {} # depth -> length -> [accs] (list to average)
    
    for f in files:
        length, depth = parse_filler_filename(f)
        if length not in allowed_lengths or depth is None:
            continue
            
        try:
            df = filter_subset(pd.read_parquet(f, columns=["dataset", "claim", "label", "pred_label"]), subset_keys)
            # Filter dataset
            if 'dataset' in df.columns:
                unique_ds = df['dataset'].unique()
                target_ds = None
                for d in unique_ds:
                    if normalize_dataset_name(d) == dataset_name:
                        target_ds = d
                        break
                
                if target_ds:
                    df_ds = df[df['dataset'] == target_ds]
                    metrics = get_metrics_for_df(df_ds)
                    if 'Acc' in metrics:
                        if depth not in data_points: data_points[depth] = {}
                        if length not in data_points[depth]: data_points[depth][length] = []
                        data_points[depth][length].append(metrics['Acc'])
        except Exception as error:
            raise ValueError(f"Cannot analyze {f}: {error}") from error
            
    stats = {}
    for d, lengths in data_points.items():
        if not lengths: continue
        # Average accs for each length
        avg_lengths = []
        for l, accs in lengths.items():
            avg_lengths.append( (l, np.mean(accs)) )
            
        avg_lengths.sort(key=lambda x: x[1])
        worst = avg_lengths[0]
        best = avg_lengths[-1]
        
        stats[d] = {'best': best, 'worst': worst}
        
    return stats

def main() -> None:
    root = project_root()
    parser = ArgumentParser(description="Generate paper tables with 0=SUPPORTS and 1=REFUTES.")
    parser.add_argument("--results-dir", type=Path, default=root / "out/results")
    parser.add_argument("--filler-results-dir", type=Path, default=None)
    parser.add_argument("--subset-file", type=Path, default=root / "data/processed/experiment_subset_seed42.pqt")
    parser.add_argument("--config", type=Path, default=root / "config/paper_config.json")
    args = parser.parse_args()
    config = load_prompt_config(args.config)
    model_names = list(config["models"])
    subset_keys = read_subset_keys(args.subset_file)
    filler_results_dir = args.filler_results_dir or args.results_dir / "subset_seed42"
    if not args.results_dir.is_dir():
        raise ValueError(f"Results directory not found: {args.results_dir}")
    print(r"%% Generated by 10_generate_latex_table.py")
    
    # ------------------------------------------------------------
    # Table 1: Base Conditions (Parametric + Contextual/Oracle)
    # The user asked for "base conditions (claim with and without evidence without context)"
    # I will add columns for "Evidence" (Contextual/Oracle)
    # ------------------------------------------------------------
    
    print(r"\begin{table*}")
    print(r"\centering")
    print(r"\caption{Parametric-only claim verification performance.}")
    print(r"\label{tab:parametric-verification-hierarchical}")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{3.5pt}")
    # Columns: Model, Dataset | Parametric (Acc, F1, P(S), P(R), R(S), R(R)) | Evidence (Acc, F1)
    # Total cols: 2 + 6 + 2 = 10
    print(r"\begin{tabular}{l l  c c cc cc c}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Dataset} & \textbf{Acc.} & \textbf{F1} & \multicolumn{2}{c}{\textbf{Precision}} & \multicolumn{2}{c}{\textbf{Recall}} & \textbf{Acc (w/ Ev.)} \\")
    print(r"\cmidrule(lr){5-6} \cmidrule(lr){7-8}")
    print(r" &  &  &  & \textbf{SUP} & \textbf{REF} & \textbf{SUP} & \textbf{REF} & \\")
    print(r"\midrule")
    
    for model in model_names:
        print(r"\midrule")
        model_name_print = model.replace("_", "-")
        model_path = os.path.join(args.results_dir, model)
        base_file = os.path.join(model_path, "res_claim_only_no_filler.pqt")
        multi_file = os.path.join(model_path, "res_multi_no_filler.pqt")
        
        has_base = os.path.exists(base_file)
        has_multi = os.path.exists(multi_file)
        
        if not has_base:
            continue
            
        df_base = filter_subset(pd.read_parquet(base_file, columns=["dataset", "claim", "label", "pred_label"]), subset_keys) if has_base else pd.DataFrame()
        df_multi = filter_subset(pd.read_parquet(multi_file, columns=["dataset", "claim", "label", "pred_label"]), subset_keys) if has_multi else pd.DataFrame()
        
        first_ds = True
        for ds in DATASETS:
            # Parametric Metrics
            m_base = {}
            if not df_base.empty and 'dataset' in df_base.columns:
                 target = next((d for d in df_base['dataset'].unique() if normalize_dataset_name(d)==ds), None)
                 if target:
                     m_base = get_metrics_for_df(df_base[df_base['dataset']==target])

            # Multi Metrics
            acc_multi = "-"
            if not df_multi.empty and 'dataset' in df_multi.columns:
                 target = next((d for d in df_multi['dataset'].unique() if normalize_dataset_name(d)==ds), None)
                 if target:
                     m_multi = get_metrics_for_df(df_multi[df_multi['dataset']==target])
                     if 'Acc' in m_multi:
                        acc_multi = f"{m_multi['Acc']:.2f}"
             
            # Format Row
            row = f"{model_name_print if first_ds else ''} & {ds}"
            
            # Unpack Base
            if m_base:
                row += f" & {m_base['Acc']:.2f} & {m_base['F1']:.2f}"
                row += f" & {m_base['Prec_SUP']:.2f} & {m_base['Prec_REF']:.2f}"
                row += f" & {m_base['Rec_SUP']:.2f} & {m_base['Rec_REF']:.2f}"
            else:
                row += " & - & - & - & - & - & -"
            
            row += f" & {acc_multi}"
                
            print(row + r" \\")
            first_ds = False
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")
    
    # ------------------------------------------------------------
    # Table 2: Depth Analysis
    # "for each depth (from 0, 10, 20, 30 to 100) the best and worst accurcay accorss the available contexts"
    # ------------------------------------------------------------
    print(r"\begin{table*}")
    print(r"\caption{Best and Worst Accuracy across Context Lengths per Depth position. Format: Acc(BestCtx)/Acc(WorstCtx)}")
    print(r"\tiny")
    print(r"\setlength{\tabcolsep}{1.5pt}")
    
    unique_depths = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Header
    # M D | D0 | D10 ...
    specs = "ll" + "c" * len(unique_depths)
    print(f"\\begin{{tabular}}{{{specs}}}")
    print(r"\toprule")
    h = r"\textbf{Model} & \textbf{DS}"
    for d in unique_depths: h += f" & \\textbf{{{d}}}"
    print(h + r" \\")
    print(r"\midrule")
    
    for model in model_names:
        print(r"\midrule")
        model_name_print = model.replace("_", "-")
        first_ds = True
        printed_any = False
        for ds in DATASETS:
            stats = get_depth_stats(model, ds, filler_results_dir, subset_keys, config["settings"]["filler_lengths"][model])
            if not stats: continue
            
            row = f"{model_name_print if first_ds else ''} & {ds}"
            
            has_data = False
            for d in unique_depths:
                if d in stats:
                     b_len, b_acc = stats[d]['best']
                     w_len, w_acc = stats[d]['worst']
                     def fl(l): return f"{int(l/1000)}k" if l>=1000 else str(l)
                     cell = f"{b_acc:.2f}({fl(b_len)}) / {w_acc:.2f}({fl(w_len)})"
                     has_data = True
                else:
                     cell = "-"
                row += f" & {cell}"
            
            if has_data:
                print(row + r" \\")
                first_ds = False
                printed_any = True
                
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate standardized communication cost vs. performance evaluation tables.
Produces:
  - results/ablations/comm_vs_performance.csv
  - results/ablations/comm_vs_performance.md
  - results/ablations/comm_vs_performance.tex (LaTeX booktabs format)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def generate_comm_performance_tables(
    csv_input_path: str = "results/ablations/comm_round_traces.csv",
    output_dir: str = "results/ablations",
    target_accuracy: float = 80.0,
) -> pd.DataFrame:
    df = pd.read_csv(csv_input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_round = df["round"].max()
    final_df = df[df["round"] == max_round]

    ref_fixed53 = 42792.0
    ref_fine104 = 71208.0
    ref_coarse32 = 28576.0

    methods_order = [
        "FedAvg-MLP",
        "FedAvg-KAN (fixed 5,3)",
        "FedProx-KAN",
        "Heuristic-KAN",
        "PPO-Adaptive-KAN",
        "Uniform-Fleet-(3,2)",
        "Uniform-Fleet-(10,4)",
        "Random-config KAN",
    ]
    # Filter to methods present
    methods = [m for m in methods_order if m in df["method"].unique()]
    alphas = sorted(df["alpha"].unique())

    rows: List[Dict[str, Any]] = []

    for alpha in alphas:
        alpha_df = df[df["alpha"] == alpha]
        final_alpha_df = final_df[final_df["alpha"] == alpha]

        for method in methods:
            m_final = final_alpha_df[final_alpha_df["method"] == method]
            m_all = alpha_df[alpha_df["method"] == method]

            if m_final.empty:
                continue

            acc_mean = m_final["test_acc"].mean()
            acc_std = m_final["test_acc"].std() if len(m_final) > 1 else 0.0

            f1_mean = m_final["test_macro_f1"].mean()
            f1_std = m_final["test_macro_f1"].std() if len(m_final) > 1 else 0.0

            bacc_mean = m_final["test_balanced_acc"].mean()
            bacc_std = m_final["test_balanced_acc"].std() if len(m_final) > 1 else 0.0

            mb_mean = m_final["cum_MB"].mean()
            mb_std = m_final["cum_MB"].std() if len(m_final) > 1 else 0.0

            params_up_mean = m_all["params_up"].mean()

            # Savings relative to baselines
            s_fixed53 = (1.0 - params_up_mean / ref_fixed53) * 100.0
            s_fine104 = (1.0 - params_up_mean / ref_fine104) * 100.0
            s_coarse32 = (1.0 - params_up_mean / ref_coarse32) * 100.0

            # Round & MB to target accuracy
            grp_by_round = m_all.groupby("round").agg({"test_acc": "mean", "cum_MB": "mean"}).reset_index()
            reached = grp_by_round[grp_by_round["test_acc"] >= target_accuracy]
            if not reached.empty:
                first_row = reached.iloc[0]
                rounds_to_target = int(first_row["round"])
                mb_to_target = float(first_row["cum_MB"])
            else:
                rounds_to_target = None
                mb_to_target = None

            rows.append({
                "method": method,
                "alpha": alpha,
                "final_acc_mean": acc_mean,
                "final_acc_std": acc_std,
                "final_macro_f1_mean": f1_mean,
                "final_macro_f1_std": f1_std,
                "final_balanced_acc_mean": bacc_mean,
                "final_balanced_acc_std": bacc_std,
                "total_MB_mean": mb_mean,
                "total_MB_std": mb_std,
                "params_per_round": int(round(params_up_mean)),
                "savings_vs_fixed53_pct": s_fixed53,
                "savings_vs_fine104_pct": s_fine104,
                "savings_vs_coarse32_pct": s_coarse32,
                "rounds_to_target_acc": rounds_to_target,
                "mb_to_target_acc": mb_to_target,
            })

    summary_df = pd.DataFrame(rows)

    # 1. Export CSV
    csv_file = out_dir / "comm_vs_performance.csv"
    summary_df.to_csv(csv_file, index=False)
    print(f"Saved CSV table to {csv_file}")

    # 2. Export Markdown Table
    md_file = out_dir / "comm_vs_performance.md"
    with open(md_file, "w") as f:
        f.write("# Federated Arrhythmia Classification: Communication Cost vs. Performance\n\n")
        f.write(f"Evaluated on MIT-BIH ECG (AAMI EC57 standard, inter-patient split, 8 clients, 5 seeds 42–46, 10 rounds).\n")
        f.write(f"Target accuracy threshold: ≥ {target_accuracy:.0f}% test accuracy.\n\n")

        for alpha in alphas:
            f.write(f"### Non-IID Dirichlet Skew α = {alpha}\n\n")
            f.write("| Method | Test Acc (%) | Macro F1 (%) | Balanced Acc (%) | Params / Round | Total MB | Savings vs (5,3) | Savings vs (10,4) | Comm to 80% Acc |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

            sub_alpha = summary_df[summary_df["alpha"] == alpha]
            for _, r in sub_alpha.iterrows():
                target_str = f"{r['rounds_to_target_acc']} rds ({r['mb_to_target_acc']:.2f} MB)" if pd.notna(r["rounds_to_target_acc"]) else "Not Reached"
                f.write(
                    f"| **{r['method']}** | "
                    f"{r['final_acc_mean']:.2f} ± {r['final_acc_std']:.2f} | "
                    f"{r['final_macro_f1_mean']:.2f} ± {r['final_macro_f1_std']:.2f} | "
                    f"{r['final_balanced_acc_mean']:.2f} ± {r['final_balanced_acc_std']:.2f} | "
                    f"{r['params_per_round']:,} | "
                    f"{r['total_MB_mean']:.2f} ± {r['total_MB_std']:.2f} | "
                    f"{r['savings_vs_fixed53_pct']:+.1f}% | "
                    f"{r['savings_vs_fine104_pct']:+.1f}% | "
                    f"{target_str} |\n"
                )
            f.write("\n")
    print(f"Saved Markdown table to {md_file}")

    # 3. Export LaTeX booktabs Table
    tex_file = out_dir / "comm_vs_performance.tex"
    with open(tex_file, "w") as f:
        f.write("% Auto-generated by experiments/generate_comm_tables.py\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Arrhythmia Detection Performance vs. Communication Overhead across Dirichlet non-IID Skews ($\\\\alpha \\\\in \\{0.1, 0.5, 1.0\\}$) on MIT-BIH (AAMI EC57, inter-patient split, 5 seeds, 10 rounds).}\n")
        f.write("\\label{tab:comm_vs_performance}\n")
        f.write("\\begin{tabular}{llcccccc}\n")
        f.write("\\toprule\n")
        f.write("$\\alpha$ & Method & Test Acc (\\%) & Macro F1 (\\%) & Balanced Acc (\\%) & Params/Rnd & Total Comm (MB) & Savings vs (5,3) \\\\\n")
        f.write("\\midrule\n")

        for a_idx, alpha in enumerate(alphas):
            sub_alpha = summary_df[summary_df["alpha"] == alpha]
            for m_idx, (_, r) in enumerate(sub_alpha.iterrows()):
                alpha_col = f"\\multirow{{{len(sub_alpha)}}}{{*}}{{{alpha}}}" if m_idx == 0 else ""
                clean_method = r["method"].replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
                bold_start = "\\textbf{" if "PPO" in r["method"] else ""
                bold_end = "}" if "PPO" in r["method"] else ""

                f.write(
                    f"{alpha_col} & {bold_start}{clean_method}{bold_end} & "
                    f"{r['final_acc_mean']:.2f} $\\pm$ {r['final_acc_std']:.2f} & "
                    f"{r['final_macro_f1_mean']:.2f} $\\pm$ {r['final_macro_f1_std']:.2f} & "
                    f"{r['final_balanced_acc_mean']:.2f} $\\pm$ {r['final_balanced_acc_std']:.2f} & "
                    f"{r['params_per_round']:,} & "
                    f"{r['total_MB_mean']:.2f} & "
                    f"{r['savings_vs_fixed53_pct']:+.1f}\\% \\\\\n"
                )
            if a_idx < len(alphas) - 1:
                f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    print(f"Saved LaTeX table to {tex_file}")

    return summary_df


if __name__ == "__main__":
    generate_comm_performance_tables()

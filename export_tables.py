import os

def generate_paper_tables(experiments_data):
    """
    输入不同模型配置下的实验数据，一键导出 Markdown 和 LaTeX 三线表代码
    """
    # ----------------------------------------------------
    # 1. 导出 Markdown 对比表格 (供本地快速预览与文档归档)
    # ----------------------------------------------------
    md_table = "### 📊 论文实验结果对比表 (Markdown 格式)\n\n"
    md_table += "| Method | Modality Isolation | $\\lambda_{ortho}$ | MME Perception | MME Cognition | MME Total | MMBench Dev (%) |\n"
    md_table += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"

    for row in experiments_data:
        md_table += f"| {row['method']} | {row['isolation']} | {row['lambda']} | {row['mme_p']:.1f} | {row['mme_c']:.1f} | **{row['mme_total']:.1f}** | **{row['mmbench']:.1f}** |\n"

    # ----------------------------------------------------
    # 2. 导出 LaTeX 标准三线表代码 (用于直接复制粘贴到 Overleaf 论文源码)
    # ----------------------------------------------------
    latex_table = "% ==========================================================\n"
    latex_table += "%  Copy & Paste the following LaTeX code into your paper!   \n"
    latex_table += "% ==========================================================\n"
    latex_table += "\\begin{table}[t]\n"
    latex_table += "  \\centering\n"
    latex_table += "  \\caption{Performance comparison on MME and MMBench benchmarks. The best results are highlighted in \\textbf{bold}.}\n"
    latex_table += "  \\label{tab:main_results}\n"
    latex_table += "  \\begin{tabular}{lccccc}\n"
    latex_table += "    \\toprule\n"
    latex_table += "    \\textbf{Method} & \\textbf{Iso.} & $\\lambda_{ortho}$ & \\textbf{MME (P)} & \\textbf{MME (C)} & \\textbf{MMBench} \\\\\n"
    latex_table += "    \\midrule\n"

    for row in experiments_data:
        is_best = row.get("is_best", False)
        if is_best:
            latex_table += f"    \\textbf{{{row['method']}}} & {row['isolation']} & {row['lambda']} & \\textbf{{{row['mme_p']:.1f}}} & \\textbf{{{row['mme_c']:.1f}}} & \\textbf{{{row['mmbench']:.1f}}} \\\\\n"
        else:
            latex_table += f"    {row['method']} & {row['isolation']} & {row['lambda']} & {row['mme_p']:.1f} & {row['mme_c']:.1f} & {row['mmbench']:.1f} \\\\\n"

    latex_table += "    \\bottomrule\n"
    latex_table += "  \\end{tabular}\n"
    latex_table += "\\end{table}\n"

    # 输出打印并保存为文件
    print(md_table)
    print(latex_table)

    with open("results_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
    print("💾 LaTeX 表格源码已成功保存至 ./results_table.tex 文件！")


if __name__ == "__main__":
    # 模拟填入你的消融实验/对比实验数据
    mock_experiments = [
        {"method": "LLaVA-1.5-7B (Standard)", "isolation": "No", "lambda": "0.00", "mme_p": 1210.0, "mme_c": 380.0, "mme_total": 1590.0, "mmbench": 64.3, "is_best": False},
        {"method": "MoE Baseline (Soft Route)", "isolation": "No", "lambda": "0.00", "mme_p": 1235.0, "mme_c": 405.0, "mme_total": 1640.0, "mmbench": 67.1, "is_best": False},
        {"method": "OrthoEvo-MoE (w/o Ortho)", "isolation": "Yes", "lambda": "0.00", "mme_p": 1255.0, "mme_c": 420.0, "mme_total": 1675.0, "mmbench": 69.8, "is_best": False},
        {"method": "OrthoEvo-MoE (Ours Full)", "isolation": "Yes", "lambda": "0.05", "mme_p": 1285.5, "mme_c": 445.0, "mme_total": 1730.5, "mmbench": 72.4, "is_best": True},
    ]

    generate_paper_tables(mock_experiments)
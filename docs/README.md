# Docs — Results & Discussion

| File | Purpose |
| --- | --- |
| `Results_and_Discussion.md` | Markdown report (tables, discussion, figure captions) |
| `Results_and_Discussion.docx` | Word report with embedded figures |
| `aggregate_results.py` | Collect mean±std metrics / runtimes from `../results` |
| `visualize_metrics.py` | Static heat-coded metric tables + AUROC / label-efficiency plots |
| `visualize_runtime_complexity.py` | Runtime & parameter tables/plots |
| `generate_word_report.py` | Build the `.docx` from aggregated data + figures |
| `data/` | CSV / JSON aggregates |
| `figures/` | PNG figures |

```powershell
python docs/aggregate_results.py
python docs/visualize_metrics.py
python docs/visualize_runtime_complexity.py
python docs/generate_word_report.py
```

Requires: `pandas`, `numpy`, `matplotlib`, `python-docx`.

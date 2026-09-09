import matplotlib.pyplot as plt

metrics = {'Precision': 0.429, 'Recall': 1.0, 'F1': 0.6}
plt.figure(figsize=(3.5, 2.5))
plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'red'])
plt.ylim(0, 1.1)
plt.ylabel("Score", fontsize=9)
plt.title("AutoMPG Causal Discovery", fontsize=9)
plt.grid(axis='y', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.savefig("autompg_results.pdf", format='pdf')
plt.savefig("autompg_results.png", dpi=300)
print("Saved autompg_results.pdf and autompg_results.png")
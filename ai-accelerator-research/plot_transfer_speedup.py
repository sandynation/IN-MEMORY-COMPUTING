import matplotlib.pyplot as plt

speedups = [4.27, 4.20, 3.99, 4.17, 3.98, 3.30, 4.27, 4.01, 4.06, 3.95]

plt.figure(figsize=(3.5, 2.5))
plt.boxplot(speedups, vert=True, widths=0.5)
plt.ylabel("Speedup (x)", fontsize=9)
plt.title("Transfer Learning Speedup (10 trials)", fontsize=9)
plt.xticks([1], ["NCPG"])
plt.grid(axis='y', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.savefig("transfer_speedup.pdf", format='pdf')
plt.savefig("transfer_speedup.png", dpi=300)
print("Saved transfer_speedup.pdf and transfer_speedup.png")
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("scalability.csv")  # expects columns: num_variables, training_time_seconds

plt.figure(figsize=(3.5, 2.5))
plt.plot(df["num_variables"], df["training_time_seconds"], 'o-', color='black', linewidth=1.5, markersize=5)
plt.xlabel("Number of Input Variables", fontsize=9)
plt.ylabel("Training Time (seconds)", fontsize=9)
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.savefig("scalability.pdf", format='pdf')
plt.savefig("scalability.png", dpi=300)
print("Saved scalability.pdf and scalability.png")
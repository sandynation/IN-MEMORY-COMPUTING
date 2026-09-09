import matplotlib.pyplot as plt

plt.figure(figsize=(2.5, 2.5))
plt.bar(["Passed", "Failed"], [100, 0], color=['green', 'red'])
plt.ylabel("Percentage (%)", fontsize=9)
plt.title("Synthesis Pass Rate (n=100)", fontsize=9)
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig("synthesis_passrate.pdf", format='pdf')
plt.savefig("synthesis_passrate.png", dpi=300)
print("Saved synthesis_passrate.pdf and synthesis_passrate.png")
import json

import matplotlib.pyplot as plt


with open("artifacts/experiment.json") as f:
    data = json.load(f)

plt.plot(range(1, len(data["residual_history"]) + 1),
         data["residual_history"])

plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Observed-entry residual")
plt.title("SVT Convergence")
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("artifacts/convergence.png", dpi=150)
plt.show()

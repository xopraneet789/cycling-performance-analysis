import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.factorplots import interaction_plot

# ---------- 1. LOAD DATA ----------

df = pd.read_csv(
    "cycling.txt",
    sep=None,          # let pandas infer delimiter
    engine="python"
)

print(df.head())
print(df.columns)

# Just to be explicit:
# columns: all_riders, rider_class, stage, points, stage_class


# ---------- 2. GLOBAL PLOTTING STYLE ----------

sns.set(style="whitegrid")


# ---------- 3. FIGURE 1: Boxplot by Rider Class ----------

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="rider_class", y="points")
plt.title("Points Distribution by Rider Class")
plt.xlabel("Rider Class")
plt.ylabel("Points")
plt.tight_layout()
plt.savefig("figure1_boxplot_rider_class.png", dpi=300)
plt.show()


# ---------- 4. FIGURE 2: Boxplot by Stage Class × Rider Class ----------

plt.figure(figsize=(9, 6))
sns.boxplot(data=df, x="stage_class", y="points", hue="rider_class")
plt.title("Points by Rider Class Across Stage Types")
plt.xlabel("Stage Class")
plt.ylabel("Points")
plt.legend(title="Rider Class")
plt.tight_layout()
plt.savefig("figure2_boxplot_stage_rider.png", dpi=300)
plt.show()


# ---------- 5. FIGURE 3: Interaction Plot (Rider Class × Stage Class) ----------

plt.figure(figsize=(9, 6))
interaction_plot(
    df["stage_class"],
    df["rider_class"],
    df["points"],
    markers=["o", "s", "D", "^"],
    ms=6
)
plt.title("Interaction Effect: Rider Class and Stage Type")
plt.xlabel("Stage Class")
plt.ylabel("Mean Points")
plt.tight_layout()
plt.savefig("figure3_interaction.png", dpi=300)
plt.show()


# ---------- 6. FIGURE 4: Histogram of Points ----------

plt.figure(figsize=(8, 6))
sns.histplot(df["points"], bins=20, kde=True)
plt.title("Distribution of Points")
plt.xlabel("Points")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("figure4_histogram_points.png", dpi=300)
plt.show()

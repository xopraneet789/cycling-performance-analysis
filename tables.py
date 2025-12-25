import pandas as pd
import numpy as np

from scipy.stats import f_oneway, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# -------------------------
# 1. LOAD DATA
# -------------------------

df = pd.read_csv(
    "cycling.txt",
    sep=None,
    engine="python"
)

print("Columns:", df.columns.tolist())
# Expected: ['all_riders', 'rider_class', 'stage', 'points', 'stage_class']

# Make sure these are categorical for ANOVA models
df["rider_class"] = df["rider_class"].astype("category")
df["stage_class"] = df["stage_class"].astype("category")


# -------------------------
# 2. DESCRIPTIVE TABLES
# -------------------------

# Table 1: Descriptive stats by rider class
table1 = df.groupby("rider_class")["points"].agg(
    N = "count",
    Mean = "mean",
    SD = "std",
    Median = "median",
    Q1 = lambda x: x.quantile(0.25),
    Q3 = lambda x: x.quantile(0.75)
).reset_index()

table1.to_csv("table1_descriptive_rider_class.csv", index=False)

# Table 2: Descriptive stats by stage class
table2 = df.groupby("stage_class")["points"].agg(
    N = "count",
    Mean = "mean",
    SD = "std",
    Median = "median",
    Q1 = lambda x: x.quantile(0.25),
    Q3 = lambda x: x.quantile(0.75)
).reset_index()

table2.to_csv("table2_descriptive_stage_class.csv", index=False)

# Table 7: Descriptive stats by rider class × stage class
table7 = df.groupby(["rider_class", "stage_class"])["points"].agg(
    N = "count",
    Mean = "mean",
    SD = "std",
    Median = "median",
    Q1 = lambda x: x.quantile(0.25),
    Q3 = lambda x: x.quantile(0.75)
).reset_index()

table7.to_csv("table7_descriptive_rider_stage.csv", index=False)


# -------------------------
# 3. ONE-WAY ANOVA (RIDER CLASS)
# -------------------------

groups = [g["points"].values for _, g in df.groupby("rider_class")]
anova_res = f_oneway(*groups)

# Prepare Table 3 (APA style)
table3 = pd.DataFrame({
    "Source": ["Rider class", "Within"],
    "df": [len(groups) - 1, len(df) - len(groups)],
    "F": [anova_res.statistic, np.nan],
    "p": [anova_res.pvalue, np.nan]
})
table3.to_csv("table3_one_way_anova.csv", index=False)


# -------------------------
# 4. KRUSKAL–WALLIS TEST
# -------------------------

kw_res = kruskal(*groups)

table4 = pd.DataFrame({
    "Test": ["Kruskal-Wallis H"],
    "df": [len(groups) - 1],
    "H": [kw_res.statistic],
    "p": [kw_res.pvalue]
})
table4.to_csv("table4_kruskal_wallis.csv", index=False)


# -------------------------
# 5. TWO-WAY ANOVA (RIDER × STAGE)
# -------------------------

model = ols("points ~ C(rider_class) + C(stage_class) + C(rider_class):C(stage_class)", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Rename columns for APA style
table5 = anova_table.reset_index().rename(columns={
    "index": "Source",
    "sum_sq": "SS",
    "df": "df",
    "F": "F",
    "PR(>F)": "p"
})
table5.to_csv("table5_two_way_anova.csv", index=False)


# -------------------------
# 6. POST-HOC COMPARISONS (Tukey HSD)
# -------------------------

tukey = pairwise_tukeyhsd(endog=df["points"], groups=df["rider_class"], alpha=0.05)
tukey_df = pd.DataFrame(data=tukey.summary()[1:], columns=tukey.summary()[0])

# Clean column names to something nicer
tukey_df.columns = ["group1", "group2", "mean_diff", "p_adj", "lower", "upper", "reject"]
tukey_df.to_csv("table6_posthoc_tukey.csv", index=False)


# -------------------------
# 7. SUMMARY TABLE ANOVA + KW (Table 8)
# -------------------------

table8 = pd.DataFrame({
    "Test": ["One-way ANOVA", "Kruskal-Wallis"],
    "Statistic": [anova_res.statistic, kw_res.statistic],
    "df": [f"{len(groups)-1}, {len(df)-len(groups)}", len(groups)-1],
    "p": [anova_res.pvalue, kw_res.pvalue]
})
table8.to_csv("table8_anova_kw_summary.csv", index=False)

print("All tables saved as CSV.")

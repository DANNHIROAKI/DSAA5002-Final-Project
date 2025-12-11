"""Cluster profiling visualizations."""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_income_vs_spent(df: pd.DataFrame, cluster_labels: pd.Series):
    """Create a scatter plot of income vs spending colored by cluster."""
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Income", y="monetary", hue=cluster_labels, palette="tab10", ax=ax)
    ax.set_title("Income vs Monetary by Cluster")
    return fig

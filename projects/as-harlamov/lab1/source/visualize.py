import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize():
    with open("../results/ollama_experiments.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data)

    df["model_short"] = df["model"].str.split(":").str[0]

    df["input_tokens"] = pd.to_numeric(df["input_tokens"], errors="coerce")
    df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="model_short", y="latency_ms", hue="mode", errorbar=None)
    plt.title("Среднее время отклика (мс) по модели и режиму")
    plt.ylabel("Время (мс)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("../results/latency_by_model_mode.png", dpi=150)
    plt.show()

    mode_summary = df.groupby("mode")[["latency_ms", "output_tokens"]].mean().reset_index()
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel("Режим")
    ax1.set_ylabel("Время отклика (мс)", color=color)
    ax1.bar(mode_summary["mode"], mode_summary["latency_ms"], color=color, alpha=0.6, label="Latency")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Средняя длина ответа (токены)", color=color)
    ax2.plot(
        mode_summary["mode"],
        mode_summary["output_tokens"],
        color=color,
        marker="o",
        linewidth=2,
        label="Output tokens",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    summary = df.groupby(["model_short", "prompt", "mode"]).agg(
        avg_latency=("latency_ms", "mean"),
        avg_output_tokens=("output_tokens", "mean"),
    ).round(2)

    print("\nСводная таблица:")
    print(summary)

    summary.to_csv("../results/summary_table.csv")


if __name__ == '__main__':
    visualize()

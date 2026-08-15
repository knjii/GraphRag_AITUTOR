from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "report_assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def build_mineru_example() -> None:
    source = ROOT / "documents" / "lab_1_ITMO_matmod-2.png"
    image = Image.open(source)

    fig, ax = plt.subplots(figsize=(10, 6.3), dpi=160)
    ax.imshow(image)
    ax.set_xlim(80, 1110)
    ax.set_ylim(1150, 90)
    ax.axis("off")

    blocks = [
        ((355, 150, 645, 555), "#d8ffae", "1"),
        ((430, 775, 510, 58), "#bcdcff", "2"),
        ((130, 875, 430, 65), "#d8c5ff", "3"),
        ((130, 950, 910, 105), "#efb5ca", "4"),
        ((430, 1070, 460, 70), "#c9ffb5", "5"),
    ]
    for (x, y, w, h), color, number in blocks:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor=color,
                edgecolor="none",
                alpha=0.42,
            )
        )
        ax.text(
            x + w + 8,
            y + 18,
            number,
            color="#e84b68",
            fontsize=12,
            fontweight="bold",
        )

    ax.add_patch(
        Rectangle(
            (85, 95),
            1015,
            1040,
            fill=False,
            edgecolor="#718096",
            linewidth=1.1,
            linestyle=(0, (3, 2)),
        )
    )
    fig.tight_layout(pad=0)
    fig.savefig(ASSETS / "mineru_layout_result.png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def horizontal_panel(ax, labels, values, title, xlabel, xlim=None) -> None:
    color = "#79b84a"
    y = list(range(len(labels)))
    ax.barh(y, values, color=color, height=0.55)
    ax.set_yticks(y, labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, fontweight="bold", color="#14286b", loc="left")
    ax.set_xlabel(xlabel, fontsize=7, color="#718096")
    ax.tick_params(axis="x", labelsize=7, colors="#718096")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    if xlim:
        ax.set_xlim(*xlim)
    for index, value in enumerate(values):
        label = f"{value:.1f}" if value > 10 else f"{value:.4f}"
        ax.text(value, index, f" {label}", va="center", fontsize=7, color="#14286b", fontweight="bold")


def build_deepeval_comparison() -> None:
    labels = [
        "qwen2.5-vl:3b / qwen3.5:4b",
        "qwen3.5:4b / qwen3.5:4b",
        "deepseek-r1:7b / qwen3.5:4b",
        "deepseek-r1:8b / qwen3.5:4b",
        "ministral3:8b / qwen3.5:4b",
    ]
    latency = [8606.2, 9331.9, 10492.0, 11054.9, 13038.3]

    relevancy_labels = [
        "deepseek-r1:7b / qwen3.5:4b",
        "ministral3:8b / qwen3.5:4b",
        "deepseek-r1:8b / qwen3.5:4b",
        "qwen3.5:4b / qwen3.5:4b",
        "qwen2.5-vl:3b / qwen3.5:4b",
    ]
    relevancy = [0.9913, 0.9669, 0.9623, 0.8919, 0.8908]

    faith_labels = [
        "deepseek-r1:8b / qwen3.5:4b",
        "ministral3:8b / qwen3.5:4b",
        "deepseek-r1:7b / qwen3.5:4b",
        "qwen3.5:4b / qwen3.5:4b",
        "qwen2.5-vl:3b / qwen3.5:4b",
    ]
    faithfulness = [0.7116, 0.6911, 0.6711, 0.6400, 0.5801]

    precision_labels = [
        "qwen3.5:4b / qwen3.5:4b",
        "deepseek-r1:8b / qwen3.5:4b",
        "deepseek-r1:7b / qwen3.5:4b",
        "qwen2.5-vl:3b / qwen3.5:4b",
        "ministral3:8b / qwen3.5:4b",
    ]
    precision = [0.8000, 0.7857, 0.7667, 0.7667, 0.7586]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), dpi=160)
    horizontal_panel(axes[0, 0], labels, latency, "EvaluationLatency: query model / judge model", "seconds", (0, 14500))
    horizontal_panel(axes[0, 1], relevancy_labels, relevancy, "Answer Relevancy (mean score)", "mean score [0..1]", (0, 1.05))
    horizontal_panel(axes[1, 0], faith_labels, faithfulness, "Faithfulness (mean score)", "mean score [0..1]", (0, 1.05))
    horizontal_panel(axes[1, 1], precision_labels, precision, "Contextual Precision (mean score)", "mean score [0..1]", (0, 1.05))
    fig.suptitle(
        "Сравнение query model / judge model по latency и DeepEval-метрикам",
        fontsize=12,
        color="#4a6485",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2, w_pad=2.2)
    fig.savefig(ASSETS / "deepeval_llm_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_graph_fragment() -> None:
    graph = nx.DiGraph()
    center = "Passage\n56556a…703585…"
    source_a = "Source\nDayzenrot…"
    source_b = "Source\nDayzenrot…"

    keywords = [
        "градиент", "список", "методами", "направление", "макет",
        "качество", "размер шага", "слишком", "размер", "шаг",
    ]
    entities = [
        "градиент…", "размер шага", "спуск", "макет", "качество",
        "направление", "взяв", "некорректно", "перпендикуляр…",
        "аналогичным", "методами", "слишком",
    ]

    graph.add_edge(source_a, center, relation="HAS_PASSAGE")
    graph.add_edge(center, source_b, relation="NEXT")
    for item in keywords:
        graph.add_edge(center, f"K:{item}", relation="HAS_KEYWORD")
    for item in entities:
        graph.add_edge(center, f"E:{item}", relation="MENTIONS")

    positions = {center: (0, 0), source_a: (-0.15, 0.1), source_b: (1.05, -0.15)}
    keyword_nodes = [node for node in graph if node.startswith("K:")]
    entity_nodes = [node for node in graph if node.startswith("E:")]
    for idx, node in enumerate(keyword_nodes):
        angle = 0.18 + idx * 0.49
        positions[node] = (1.25 * __import__("math").cos(angle), 1.25 * __import__("math").sin(angle))
    for idx, node in enumerate(entity_nodes):
        angle = 0.05 + idx * (2 * __import__("math").pi / len(entity_nodes))
        radius = 1.75
        positions[node] = (radius * __import__("math").cos(angle), radius * __import__("math").sin(angle))

    fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=160)
    ax.set_facecolor("#f8fafc")
    node_colors = []
    sizes = []
    labels = {}
    for node in graph.nodes:
        if node == center:
            node_colors.append("#b8b6ea")
            sizes.append(1800)
            labels[node] = node
        elif node.startswith("K:"):
            node_colors.append("#ffb74d")
            sizes.append(1250)
            labels[node] = node[2:]
        elif node.startswith("E:"):
            node_colors.append("#c98f7e")
            sizes.append(1250)
            labels[node] = node[2:]
        else:
            node_colors.append("#a98bd4")
            sizes.append(1400)
            labels[node] = node

    nx.draw_networkx_nodes(graph, positions, node_color=node_colors, node_size=sizes, ax=ax, linewidths=0.6, edgecolors="white")
    nx.draw_networkx_edges(graph, positions, ax=ax, arrows=True, arrowsize=12, edge_color="#8793a1", width=0.9, alpha=0.8)
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=7, ax=ax)
    edge_labels = {(u, v): data["relation"] for u, v, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_size=5, font_color="#697386", rotate=True, ax=ax)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ASSETS / "neo4j_graph_fragment.png", bbox_inches="tight", facecolor="#f8fafc")
    plt.close(fig)


if __name__ == "__main__":
    build_mineru_example()
    build_deepeval_comparison()
    build_graph_fragment()
    print(ASSETS)

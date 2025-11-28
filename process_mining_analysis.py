"""
Process Mining Analysis Script
Практическая работа: Анализ бизнес-процессов
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
from pathlib import Path
import argparse

# Подавляем только предупреждения о deprecated функциях matplotlib
warnings.filterwarnings('ignore', category=DeprecationWarning, module='matplotlib')


def parse_arguments():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Process mining analysis")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("input_data/running_example_broken.csv"),
        help="Путь к CSV/XES файлу с журналом событий",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Показать граф процесса интерактивно",
    )
    return parser.parse_args()


def load_event_log(source_path: Path) -> pd.DataFrame:
    """Загружает журнал событий из CSV или XES и нормализует колонки."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл {path} не найден")

    if path.suffix.lower() == ".xes":
        try:
            from pm4py.objects.log.importer.xes import importer as xes_importer
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise ImportError("Для работы с XES установите pm4py") from exc
        log = xes_importer.apply(str(path))
        records = []
        for trace in log:
            case_id = trace.attributes.get("concept:name") or trace.attributes.get("case:concept:name")
            for event in trace:
                records.append(
                    {
                        "case_id": case_id,
                        "activity": event.get("concept:name"),
                        "timestamp": event.get("time:timestamp"),
                        "resource": event.get("org:resource") or event.get("Resource"),
                        "cost": event.get("cost") or event.get("costs"),
                    }
                )
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
        column_mapping = {
            "case:concept:name": "case_id",
            "concept:name": "activity",
            "time:timestamp": "timestamp",
            "costs": "cost",
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    resource_columns = [col for col in ["resource", "org:resource", "Resource"] if col in df.columns]
    if resource_columns:
        df["resource"] = df[resource_columns[0]]
    else:
        df["resource"] = "unknown"
    for col in resource_columns:
        if col != "resource":
            df = df.drop(columns=col)

    required = {"case_id", "activity", "timestamp", "resource"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В журнале отсутствуют обязательные поля: {missing}")

    df["case_id"] = df["case_id"].astype(str)
    df["activity"] = df["activity"].astype(str)
    df["resource"] = df["resource"].fillna("unknown").astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    if "cost" in df.columns:
        df["cost"] = pd.to_numeric(df["cost"], errors="coerce")

    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    return df


def determine_boundary_activities(df: pd.DataFrame):
    """Определяет стартовые и конечные активности по журналу."""
    ordered = df.sort_values(["case_id", "timestamp"])
    first_events = ordered.groupby("case_id").first()
    last_events = ordered.groupby("case_id").last()

    start_counts = first_events["activity"].value_counts()
    end_counts = last_events["activity"].value_counts()

    return {
        "primary_start": start_counts.idxmax() if not start_counts.empty else None,
        "primary_end": end_counts.idxmax() if not end_counts.empty else None,
        "start_counts": start_counts,
        "end_counts": end_counts,
        "start_nodes": set(first_events["activity"].unique()),
        "end_nodes": set(last_events["activity"].unique()),
    }



def preprocess_data(df):
    """
    Шаг 1: Предварительная обработка данных
    """
    print("=" * 60)
    print("ШАГ 1: ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ")
    print("=" * 60)

    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

    print(f"\nОбщее количество событий: {len(df)}")
    print(f"Количество уникальных кейсов: {df['case_id'].nunique()}")
    print(f"Количество уникальных активностей: {df['activity'].nunique()}")
    print(f"Количество уникальных ресурсов: {df['resource'].nunique()}")

    if "cost" in df.columns:
        total_cost = df["cost"].sum()
        avg_cost = df["cost"].mean()
        print(f"Суммарные затраты (по событиям): {total_cost:.0f}")
        print(f"Средняя стоимость события: {avg_cost:.1f}")

    print("\nУникальные активности:")
    for i, activity in enumerate(df['activity'].unique(), 1):
        print(f"  {i}. {activity}")

    print("\nЧастота активностей:")
    activity_counts = df['activity'].value_counts()
    for activity, count in activity_counts.items():
        print(f"  {activity}: {count} раз ({count/len(df)*100:.1f}%)")

    print(f"\nПропущенные значения: {df.isnull().sum().sum()}")
    return df


def build_process_graph(df):
    """
    Шаг 2: Построение графа бизнес-процесса
    """
    print("\n" + "=" * 60)
    print("ШАГ 2: ПОСТРОЕНИЕ ГРАФА БИЗНЕС-ПРОЦЕССА")
    print("=" * 60)
    
    # Создаем направленный граф
    G = nx.DiGraph()
    
    # Словарь для подсчета переходов
    transitions = defaultdict(int)
    
    # Анализируем каждый кейс
    for case_id in df['case_id'].unique():
        case_events = df[df['case_id'] == case_id].sort_values('timestamp')
        activities = case_events['activity'].tolist()
        
        # Добавляем переходы между активностями
        for i in range(len(activities) - 1):
            source = activities[i]
            target = activities[i + 1]
            transitions[(source, target)] += 1
    
    # Добавляем рёбра в граф
    for (source, target), count in transitions.items():
        G.add_edge(source, target, weight=count)
    
    print(f"\nГраф процесса построен:")
    print(f"  Узлов (активностей): {G.number_of_nodes()}")
    print(f"  Рёбер (переходов): {G.number_of_edges()}")
    
    print("\nПереходы между активностями:")
    for (source, target), count in sorted(transitions.items(), key=lambda x: -x[1]):
        print(f"  {source} -> {target}: {count} раз")
    
    return G, transitions


def analyze_graph(G, df, transitions):
    """
    Шаг 3: Анализ графа процесса
    """
    print("\n" + "=" * 60)
    print("ШАГ 3: АНАЛИЗ ГРАФА ПРОЦЕССА")
    print("=" * 60)

    print("\n--- 3.1 УЗКИЕ МЕСТА ---")
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    print("\nСтепени узлов (входящие/исходящие):")
    for node in G.nodes():
        print(f"  {node}: вход={in_degrees.get(node, 0)}, выход={out_degrees.get(node, 0)}")

    bottleneck = max(in_degrees, key=in_degrees.get) if in_degrees else None
    if bottleneck:
        print(f"\nПотенциальное узкое место: '{bottleneck}' (входящих: {in_degrees[bottleneck]})")

    print("\n--- 3.2 ПОИСК ЦИКЛОВ ---")
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            print(f"Найдено циклов: {len(cycles)}")
            for i, cycle in enumerate(cycles, 1):
                cycle_str = " -> ".join(cycle + [cycle[0]])
                print(f"  Цикл {i}: {cycle_str}")
        else:
            print("Циклов не обнаружено")
    except nx.NetworkXError as e:
        print(f"Ошибка при поиске циклов: {e}")
        cycles = []

    print("\n--- 3.3 ОТКЛОНЕНИЯ (НЕСТАНДАРТНЫЕ ПУТИ) ---")
    case_traces = []
    for case_id in df['case_id'].unique():
        case_events = df[df['case_id'] == case_id].sort_values('timestamp')
        trace = tuple(case_events['activity'].tolist())
        case_traces.append((case_id, trace))

    trace_counts = defaultdict(list)
    for case_id, trace in case_traces:
        trace_counts[trace].append(case_id)

    sorted_traces = sorted(trace_counts.items(), key=lambda x: -len(x[1]))
    standard_trace = sorted_traces[0][0] if sorted_traces else None

    for trace, cases in sorted_traces:
        trace_str = " -> ".join(trace)
        label = " (СТАНДАРТНЫЙ)" if trace == standard_trace else " (ОТКЛОНЕНИЕ)"
        print(f"  Кейсы {cases}: {trace_str}{label}")

    print("\n--- 3.4 ОПТИМАЛЬНЫЕ ПУТИ ---")
    boundary_info = determine_boundary_activities(df)
    start_node = boundary_info.get('primary_start')
    end_node = boundary_info.get('primary_end')

    if start_node and end_node and start_node in G.nodes() and end_node in G.nodes():
        try:
            all_paths = list(nx.all_simple_paths(G, start_node, end_node))
            print(f"\nВсе возможные пути от '{start_node}' до '{end_node}':")
            for i, path in enumerate(all_paths, 1):
                path_str = " -> ".join(path)
                print(f"  Путь {i}: {path_str} (длина: {len(path)-1} шагов)")
            shortest = nx.shortest_path(G, start_node, end_node)
            print(f"\nОптимальный (кратчайший) путь: {' -> '.join(shortest)}")
        except nx.NetworkXNoPath:
            print(f"Путь от '{start_node}' до '{end_node}' не найден")
    else:
        print("Не удалось определить стартовую и конечную активности для поиска пути")

    return {
        "standard_trace": standard_trace,
        "trace_counts": trace_counts,
        "cycles": cycles,
        "bottleneck": bottleneck,
        "in_degrees": in_degrees,
        "out_degrees": out_degrees,
        "boundary_info": boundary_info,
    }


def analyze_duration(df):
    """
    Анализ длительности выполнения процесса
    """
    print("\n--- 3.5 АНАЛИЗ ДЛИТЕЛЬНОСТИ ПРОЦЕССОВ ---")

    case_durations = []
    for case_id in df['case_id'].unique():
        case_events = df[df['case_id'] == case_id].sort_values('timestamp')
        start_time = case_events['timestamp'].min()
        end_time = case_events['timestamp'].max()
        duration = (end_time - start_time).total_seconds() / 3600
        trace = tuple(case_events['activity'].tolist())
        case_durations.append({
            'case_id': case_id,
            'duration_hours': duration,
            'num_events': len(case_events),
            'trace': trace
        })

    duration_df = pd.DataFrame(case_durations)

    print("\nСтатистика длительности (в часах):")
    print(f"  Минимальная: {duration_df['duration_hours'].min():.1f} ч")
    print(f"  Максимальная: {duration_df['duration_hours'].max():.1f} ч")
    print(f"  Средняя: {duration_df['duration_hours'].mean():.1f} ч")
    print(f"  Медиана: {duration_df['duration_hours'].median():.1f} ч")

    threshold = duration_df['duration_hours'].quantile(0.75)
    long_cases = duration_df[duration_df['duration_hours'] > threshold]

    print(f"\nКейсы с длительностью выше 75 перцентиля ({threshold:.1f} ч):")
    for _, row in long_cases.iterrows():
        trace_str = " -> ".join(row['trace'])
        print(f"  Кейс {row['case_id']}: {row['duration_hours']:.1f} ч, {row['num_events']} событий")
        print(f"    Трейс: {trace_str}")

    return duration_df


def suggest_optimizations(df, analysis_results, duration_df):
    """
    Шаг 4: Предложение вариантов оптимизации
    """
    print("\n" + "=" * 60)
    print("ШАГ 4: ВАРИАНТЫ ОПТИМИЗАЦИИ")
    print("=" * 60)

    bottleneck = analysis_results.get('bottleneck')
    in_degrees = analysis_results.get('in_degrees', {})
    out_degrees = analysis_results.get('out_degrees', {})
    cycles = analysis_results.get('cycles', [])
    standard_trace = analysis_results.get('standard_trace')
    trace_counts = analysis_results.get('trace_counts', {})

    duration_threshold = duration_df['duration_hours'].quantile(0.75)
    critical_cases = duration_df[duration_df['duration_hours'] > duration_threshold]

    print("""
    На основе журнала предлагаются следующие направления улучшений:
    """)

    if bottleneck:
        print(
            f"1. Сгладить нагрузку на '{bottleneck}' (вход={in_degrees.get(bottleneck, 0)}, "
            f"выход={out_degrees.get(bottleneck, 0)}). Перераспределение задач или автоматизация"
            " сократят очередь перед этой активностью."
        )

    if cycles:
        cycle_str = " -> ".join(cycles[0] + [cycles[0][0]])
        print(
            f"2. Минимизировать повторы, связанные с циклом {cycle_str}. Улучшение качества входных данных"
            " или чек-листов позволит избегать возвратов."
        )

    if not critical_cases.empty:
        worst_case = critical_cases.sort_values('duration_hours', ascending=False).iloc[0]
        print(
            f"3. Разобрать длительные кейсы (например, {worst_case['case_id']} длительностью "
            f"{worst_case['duration_hours']:.1f} ч). Выявленные причины помогут сократить 25-й перцентиль."
        )

    deviations = [item for item in trace_counts.items() if item[0] != standard_trace]
    if deviations:
        deviations.sort(key=lambda x: -len(x[1]))
        trace, cases = deviations[0]
        trace_str = " -> ".join(trace)
        print(
            f"4. Формализовать отклоняющийся путь ({trace_str}), встречающийся у кейсов {cases}."
            " Чёткие SLA снижают вариативность процесса."
        )

    if 'cost' in df.columns and df['cost'].notna().any():
        cost_by_case = df.groupby('case_id')['cost'].sum().sort_values(ascending=False)
        top_case = cost_by_case.index[0]
        print(
            f"5. Пересмотреть дорогостоящие кейсы (например, {top_case} с затратами "
            f"{cost_by_case.iloc[0]:.0f}). Приоритизация окупит улучшения."
        )


def visualize_process(G, transitions, boundary_info, show_plot=False):
    """
    Визуализация графа процесса
    """
    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ ГРАФА")
    print("=" * 60)

    plt.figure(figsize=(14, 10))

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    start_nodes = boundary_info.get('start_nodes', set()) if boundary_info else set()
    end_nodes = boundary_info.get('end_nodes', set()) if boundary_info else set()
    node_colors = []
    for node in G.nodes():
        if node in start_nodes:
            node_colors.append('lightgreen')
        elif node in end_nodes:
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightblue')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9)

    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 + (w / max_weight) * 4 for w in edge_weights]

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.6,
        edge_color='gray',
        connectionstyle='arc3,rad=0.1',
        arrows=True,
        arrowsize=20,
    )

    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

    plt.title('Граф бизнес-процесса\n(цвета: старт/промежуточные/конечные)', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('process_graph.png', dpi=150, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    print("Граф сохранён в файл: process_graph.png")


def main():
    """
    Основная функция запуска анализа
    """
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("АНАЛИЗ БИЗНЕС-ПРОЦЕССА")
    print("=" * 60)
    print(f"Источник данных: {args.source}")

    df = load_event_log(args.source)
    df = preprocess_data(df)
    G, transitions = build_process_graph(df)
    analysis_results = analyze_graph(G, df, transitions)
    duration_df = analyze_duration(df)
    suggest_optimizations(df, analysis_results, duration_df)
    visualize_process(G, transitions, analysis_results.get('boundary_info'), show_plot=args.show_plot)

    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 60)

    return df, G, transitions, duration_df


if __name__ == "__main__":
    main()

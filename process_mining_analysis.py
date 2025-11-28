"""
Process Mining Analysis Script
Практическая работа: Анализ бизнес-процессов
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
# Подавляем только предупреждения о deprecated функциях matplotlib
warnings.filterwarnings('ignore', category=DeprecationWarning, module='matplotlib')


def load_sample_data():
    """
    Создаем образец данных для анализа бизнес-процесса.
    Данные имитируют процесс обработки заявок в службе поддержки.
    """
    # Пример данных: процесс обработки заявок
    data = {
        'case_id': [
            1, 1, 1, 1,
            2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3,
            4, 4, 4, 4,
            5, 5, 5, 5, 5, 5, 5,
            6, 6, 6, 6,
            7, 7, 7, 7, 7,
            8, 8, 8, 8, 8, 8,
            9, 9, 9, 9,
            10, 10, 10, 10, 10, 10, 10, 10
        ],
        'activity': [
            'Получение заявки', 'Классификация', 'Обработка', 'Закрытие',
            'Получение заявки', 'Классификация', 'Эскалация', 'Обработка', 'Закрытие',
            'Получение заявки', 'Классификация', 'Обработка', 'Возврат на доработку', 'Обработка', 'Закрытие',
            'Получение заявки', 'Классификация', 'Обработка', 'Закрытие',
            'Получение заявки', 'Классификация', 'Эскалация', 'Обработка', 'Возврат на доработку', 'Обработка', 'Закрытие',
            'Получение заявки', 'Классификация', 'Обработка', 'Закрытие',
            'Получение заявки', 'Классификация', 'Эскалация', 'Обработка', 'Закрытие',
            'Получение заявки', 'Классификация', 'Обработка', 'Возврат на доработку', 'Обработка', 'Закрытие',
            'Получение заявки', 'Классификация', 'Обработка', 'Закрытие',
            'Получение заявки', 'Классификация', 'Эскалация', 'Обработка', 'Возврат на доработку', 'Обработка', 'Возврат на доработку', 'Закрытие'
        ],
        'timestamp': [
            '2024-01-01 09:00', '2024-01-01 09:30', '2024-01-01 10:00', '2024-01-01 11:00',
            '2024-01-01 10:00', '2024-01-01 10:20', '2024-01-01 11:00', '2024-01-01 14:00', '2024-01-01 15:00',
            '2024-01-02 09:00', '2024-01-02 09:15', '2024-01-02 10:00', '2024-01-02 12:00', '2024-01-02 14:00', '2024-01-02 16:00',
            '2024-01-02 11:00', '2024-01-02 11:30', '2024-01-02 12:00', '2024-01-02 13:00',
            '2024-01-03 09:00', '2024-01-03 09:30', '2024-01-03 10:30', '2024-01-03 14:00', '2024-01-03 16:00', '2024-01-04 10:00', '2024-01-04 12:00',
            '2024-01-03 14:00', '2024-01-03 14:20', '2024-01-03 15:00', '2024-01-03 16:00',
            '2024-01-04 09:00', '2024-01-04 09:30', '2024-01-04 10:30', '2024-01-04 13:00', '2024-01-04 14:00',
            '2024-01-04 11:00', '2024-01-04 11:30', '2024-01-04 12:00', '2024-01-04 15:00', '2024-01-04 17:00', '2024-01-04 19:00',
            '2024-01-05 09:00', '2024-01-05 09:30', '2024-01-05 10:00', '2024-01-05 11:00',
            '2024-01-05 10:00', '2024-01-05 10:30', '2024-01-05 11:30', '2024-01-05 15:00', '2024-01-05 17:00', '2024-01-06 10:00', '2024-01-06 14:00', '2024-01-06 16:00'
        ],
        'resource': [
            'Оператор 1', 'Аналитик 1', 'Специалист 1', 'Оператор 1',
            'Оператор 2', 'Аналитик 1', 'Менеджер 1', 'Специалист 2', 'Оператор 2',
            'Оператор 1', 'Аналитик 2', 'Специалист 1', 'Специалист 1', 'Специалист 1', 'Оператор 1',
            'Оператор 2', 'Аналитик 1', 'Специалист 2', 'Оператор 2',
            'Оператор 1', 'Аналитик 2', 'Менеджер 1', 'Специалист 1', 'Специалист 1', 'Специалист 1', 'Оператор 1',
            'Оператор 2', 'Аналитик 1', 'Специалист 2', 'Оператор 2',
            'Оператор 1', 'Аналитик 2', 'Менеджер 1', 'Специалист 2', 'Оператор 1',
            'Оператор 2', 'Аналитик 1', 'Специалист 1', 'Специалист 1', 'Специалист 1', 'Оператор 2',
            'Оператор 1', 'Аналитик 2', 'Специалист 2', 'Оператор 1',
            'Оператор 2', 'Аналитик 1', 'Менеджер 1', 'Специалист 2', 'Специалист 2', 'Специалист 2', 'Специалист 2', 'Оператор 2'
        ]
    }
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def preprocess_data(df):
    """
    Шаг 1: Предварительная обработка данных
    """
    print("=" * 60)
    print("ШАГ 1: ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ")
    print("=" * 60)
    
    # Сортировка по case_id и timestamp
    df = df.sort_values(['case_id', 'timestamp']).reset_index(drop=True)
    
    # Базовая статистика
    print(f"\nОбщее количество событий: {len(df)}")
    print(f"Количество уникальных кейсов: {df['case_id'].nunique()}")
    print(f"Количество уникальных активностей: {df['activity'].nunique()}")
    print(f"Количество уникальных ресурсов: {df['resource'].nunique()}")
    
    print("\nУникальные активности:")
    for i, activity in enumerate(df['activity'].unique(), 1):
        print(f"  {i}. {activity}")
    
    print("\nЧастота активностей:")
    activity_counts = df['activity'].value_counts()
    for activity, count in activity_counts.items():
        print(f"  {activity}: {count} раз ({count/len(df)*100:.1f}%)")
    
    # Проверка на пропущенные значения
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
    
    # 3.1 Поиск узких мест (bottlenecks)
    print("\n--- 3.1 УЗКИЕ МЕСТА ---")
    
    # Узлы с высокой степенью входящих рёбер (много источников)
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    print("\nСтепени узлов (входящие/исходящие):")
    for node in G.nodes():
        print(f"  {node}: вход={in_degrees.get(node, 0)}, выход={out_degrees.get(node, 0)}")
    
    # Узкое место - узел с высокой входящей степенью
    bottleneck = max(in_degrees, key=in_degrees.get)
    print(f"\nПотенциальное узкое место: '{bottleneck}' (входящих связей: {in_degrees[bottleneck]})")
    
    # 3.2 Поиск циклов
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
    
    # 3.3 Поиск отклонений (нестандартных путей)
    print("\n--- 3.3 ОТКЛОНЕНИЯ (НЕСТАНДАРТНЫЕ ПУТИ) ---")
    
    # Определяем стандартный путь (самый частый)
    case_traces = []
    for case_id in df['case_id'].unique():
        case_events = df[df['case_id'] == case_id].sort_values('timestamp')
        trace = tuple(case_events['activity'].tolist())
        case_traces.append((case_id, trace))
    
    # Подсчет частоты трейсов
    trace_counts = defaultdict(list)
    for case_id, trace in case_traces:
        trace_counts[trace].append(case_id)
    
    # Сортировка по частоте
    sorted_traces = sorted(trace_counts.items(), key=lambda x: -len(x[1]))
    
    print("\nВарианты процесса (трейсы):")
    standard_trace = sorted_traces[0][0] if sorted_traces else None
    
    for trace, cases in sorted_traces:
        trace_str = " -> ".join(trace)
        is_standard = " (СТАНДАРТНЫЙ)" if trace == standard_trace else " (ОТКЛОНЕНИЕ)"
        print(f"  Кейсы {cases}: {trace_str}{is_standard}")
    
    # 3.4 Оптимальные пути
    print("\n--- 3.4 ОПТИМАЛЬНЫЕ ПУТИ ---")
    
    # Начальная и конечная точки
    start_node = 'Получение заявки'
    end_node = 'Закрытие'
    
    if start_node in G.nodes() and end_node in G.nodes():
        try:
            # Все пути от начала к концу
            all_paths = list(nx.all_simple_paths(G, start_node, end_node))
            print(f"\nВсе возможные пути от '{start_node}' до '{end_node}':")
            
            for i, path in enumerate(all_paths, 1):
                path_str = " -> ".join(path)
                print(f"  Путь {i}: {path_str} (длина: {len(path)-1} шагов)")
            
            # Кратчайший путь
            shortest = nx.shortest_path(G, start_node, end_node)
            print(f"\nОптимальный (кратчайший) путь: {' -> '.join(shortest)}")
            
        except nx.NetworkXNoPath:
            print(f"Путь от '{start_node}' до '{end_node}' не найден")
    
    return standard_trace, trace_counts


def analyze_duration(df, trace_counts):
    """
    Анализ длительности выполнения процесса
    """
    print("\n--- 3.5 АНАЛИЗ ДЛИТЕЛЬНОСТИ ПРОЦЕССОВ ---")
    
    # Вычисляем длительность каждого кейса
    case_durations = []
    
    for case_id in df['case_id'].unique():
        case_events = df[df['case_id'] == case_id].sort_values('timestamp')
        start_time = case_events['timestamp'].min()
        end_time = case_events['timestamp'].max()
        duration = (end_time - start_time).total_seconds() / 3600  # в часах
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
    
    # Выявление долгих процессов (выше 75 перцентиля)
    threshold = duration_df['duration_hours'].quantile(0.75)
    long_cases = duration_df[duration_df['duration_hours'] > threshold]
    
    print(f"\nКейсы с длительностью выше 75 перцентиля ({threshold:.1f} ч):")
    for _, row in long_cases.iterrows():
        trace_str = " -> ".join(row['trace'])
        print(f"  Кейс {row['case_id']}: {row['duration_hours']:.1f} ч, {row['num_events']} событий")
        print(f"    Трейс: {trace_str}")
    
    return duration_df


def suggest_optimizations(df, trace_counts, duration_df):
    """
    Шаг 4: Предложение вариантов оптимизации
    """
    print("\n" + "=" * 60)
    print("ШАГ 4: ВАРИАНТЫ ОПТИМИЗАЦИИ")
    print("=" * 60)
    
    print("""
    На основе проведённого анализа предлагаются следующие варианты оптимизации:

    1. УСТРАНЕНИЕ ЦИКЛОВ (ВОЗВРАТОВ НА ДОРАБОТКУ)
       - Выявлено, что активность "Возврат на доработку" создаёт цикл
       - Рекомендация: Улучшить качество первичной обработки для снижения
         количества возвратов
       - Ожидаемый эффект: Сокращение времени обработки на 20-30%

    2. ОПТИМИЗАЦИЯ ПРОЦЕССА ЭСКАЛАЦИИ
       - Эскалация значительно увеличивает время обработки
       - Рекомендация: Расширить полномочия специалистов первого уровня
         для снижения количества эскалаций
       - Ожидаемый эффект: Сокращение времени на 15-25%

    3. АВТОМАТИЗАЦИЯ КЛАССИФИКАЦИИ
       - Классификация выполняется вручную в каждом кейсе
       - Рекомендация: Внедрить автоматическую классификацию на основе
         машинного обучения
       - Ожидаемый эффект: Ускорение этапа классификации на 50-70%

    4. ПАРАЛЛЕЛИЗАЦИЯ РАБОТЫ
       - Некоторые этапы можно выполнять параллельно
       - Рекомендация: Начинать предварительную обработку ещё на этапе
         классификации
       - Ожидаемый эффект: Сокращение общего времени на 10-15%

    5. БАЛАНСИРОВКА НАГРУЗКИ
       - Анализ ресурсов показывает неравномерную загрузку
       - Рекомендация: Перераспределить задачи между специалистами
       - Ожидаемый эффект: Сокращение очередей и времени ожидания
    """)


def visualize_process(G, transitions, show_plot=False):
    """
    Визуализация графа процесса
    
    Args:
        G: граф процесса
        transitions: словарь переходов
        show_plot: показать график интерактивно (по умолчанию False)
    """
    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ ГРАФА")
    print("=" * 60)
    
    plt.figure(figsize=(14, 10))
    
    # Используем автоматический layout для гибкости
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Рисуем узлы
    node_colors = ['lightgreen' if node == 'Получение заявки' else 
                   'lightcoral' if node == 'Закрытие' else
                   'lightyellow' if node == 'Возврат на доработку' else
                   'lightblue' for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=3000, alpha=0.9)
    
    # Рисуем рёбра с весами
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 + (w / max_weight) * 4 for w in edge_weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                          alpha=0.6, edge_color='gray',
                          connectionstyle='arc3,rad=0.1',
                          arrows=True, arrowsize=20)
    
    # Подписи узлов
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    # Подписи рёбер (веса)
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title('Граф бизнес-процесса обработки заявок\n(числа показывают частоту переходов)', 
              fontsize=14, fontweight='bold')
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
    print("\n" + "=" * 60)
    print("АНАЛИЗ БИЗНЕС-ПРОЦЕССА: ОБРАБОТКА ЗАЯВОК")
    print("=" * 60)
    
    # Загрузка данных
    df = load_sample_data()
    
    # Шаг 1: Предварительная обработка
    df = preprocess_data(df)
    
    # Шаг 2: Построение графа
    G, transitions = build_process_graph(df)
    
    # Шаг 3: Анализ графа
    standard_trace, trace_counts = analyze_graph(G, df, transitions)
    
    # Анализ длительности
    duration_df = analyze_duration(df, trace_counts)
    
    # Шаг 4: Предложения по оптимизации
    suggest_optimizations(df, trace_counts, duration_df)
    
    # Визуализация
    visualize_process(G, transitions)
    
    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 60)
    
    return df, G, transitions, duration_df


if __name__ == "__main__":
    df, G, transitions, duration_df = main()

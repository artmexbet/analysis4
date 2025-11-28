# analysis4

## Задание
Выполнение практической работы предполагает решение следующий задач:
1. Выполнить предварительную обработку данных
2. Построить граф для выбранного бизнес-процесса
3. Провести анализ полученного графа на предмет наличия в нем узких мест, циклов, отклонений и оптимальных путей
4. Предложить варианты оптимизации выбранного бизнес-процесса

## Установка и запуск

```bash
pip install -r requirements.txt
python process_mining_analysis.py --source input_data/running_example_broken.csv
```

## Описание решения

### Журнал: running_example_broken.csv

Используется реальный журнал заявок авиакомпании (формат BPI Challenge). Скрипт автоматически подстраивается под CSV или XES и строит анализ по фактическим активностям: `register request`, `check ticket`, `examine casually`, `examine thoroughly`, `decide`, `reject request`, `pay compensation`, `reinitiate...` и т.д.

### Ключевые результаты (для running_example_broken.csv)
- Событий: 32  
- Кейсов: 6  
- Активностей: 8  
- Суммарные затраты (по событиям): 4 150  
- Стартовая активность: `register request`, финиш чаще всего `pay compensation`/`reject request`
- Узкое место: `decide` (максимум входящих переходов)
- Найден цикл: `decide -> reinitiate ... -> check ticket -> ... -> decide`
- Стандартный путь: `register request -> check ticket -> decide -> pay compensation`
- 75-й перцентиль длительности ≈ 213 ч; кейс `5` самая проблемная длительность и стоимость
- `process_graph.png` визуализирует переходы с утолщёнными рёбрами по частоте

### Варианты оптимизации
1. Сбалансировать нагрузку на активность принятия решения (`decide`) — автоматизировать правила или добавить экспертов.
2. Снизить повторные итерации (цикл `decide -> reinitiate ...`) через качественные требования к входу.
3. Разобрать долгие кейсы (особенно `5`) и внедрить контрольные точки.
4. Формализовать сложный путь с повторяющимися проверками, чтобы снизить вариативность.
5. Приоритизировать дорогостоящие кейсы — это окупит улучшения быстрее.

## Структура проекта

```
├── README.md
├── requirements.txt
├── process_mining_analysis.py
└── input_data/
    ├── running_example_broken.csv
    └── running-example.xes
```

## Ссылки на источники данных
- https://data.4tu.nl/collections/_/5065541/1
- https://www.tf-pm.org/resources/logs
- https://icpmconference.org/2020/process-discovery-contest/data-set/
- https://www.kaggle.com/code/samhomsi/process-mining/data
- https://icpmconference.org/2020/bpi-challenge/

# ИЗ № 1. Слежение за объектом на UAV123

Ерыкалов Арсений Сергеевич, Русинов Максим.

Проект написан по методичке к ИЗ № 1. `config.json` задаёт 12 учебных фрагментов полного UAV123. В `dev` шесть последовательностей и 2 884 кадра, в `test` шесть последовательностей и 2 650 кадров. Список исходных файлов каждого фрагмента записывается в `manifest.json`.

## Запуск из папки проекта

В отдельном окружении Python 3.10 установить зависимости. OpenCV Contrib включает трекеры CSRT, KCF и MOSSE.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python prepare_data.py --download --split dev
python audit.py
python run_trackers.py --split dev
python robustness.py --tune
python evaluate.py --split dev --methods CSRT KCF MOSSE LK STUDENT
python report.py --split dev
```

Для локальной копии полного архива UAV123 команда подготовки выглядит так. Для `test` используется тот же архив и `--split test`.

```powershell
python prepare_data.py --archive "D:\data\Dataset_UAV123.zip" --split dev
```

Для итоговой проверки сначала сохранить решение, выбранное по `dev`, в `results/dev/selection.json`. Команда `robustness.py --tune` создаёт этот файл и таблицу подбора. Если в комплекте уже есть `submission.json` и тестовые результаты, используй их для изучения итогового прогона.

```powershell
python prepare_data.py --download --split test
python audit.py --freeze submission.json
python run_trackers.py --split test --methods CSRT KCF MOSSE LK STUDENT --frozen submission.json
python evaluate.py --split test --methods CSRT KCF MOSSE LK STUDENT
python report.py --split test
```

## Результаты

- `results/*/predictions` содержит номер кадра, рамку в исходных координатах и время вызова `update`. Строка с `nan` означает, что трекер не выдал рамку.
- `summary.csv` усредняет метрики по шести последовательностям с одинаковым весом. Кадры в секунду вычисляются как число вызовов `update` за суммарное время этих вызовов.
- `failures.json` содержит все эпизоды IoU < 0,1. В `examples.json` перечислены эпизоды, для которых подготовлены иллюстрации. Рамки на иллюстрациях и видео окрашены в зелёный для эталона и оранжевый для ответа трекера.
- `results/dev/tuning.csv` и `robustness.csv` содержат результаты подбора порога изменения масштаба и проверок при изменении яркости и размера входных кадров.
- `submission.json` содержит выбранный параметр, контрольные суммы кода, разметки и кадров. Аудит сверяет состав и длину фрагментов и проверяет отсутствие файлов с одинаковым SHA-256 в разных частях.
- В комплекте результатов `results/manifests` содержит списки исходных кадров обоих фрагментов эксперимента. Для повторного запуска восстанови кадры командой `prepare_data.py`.

Источник данных — [официальный архив UAV123 и UAV20L](https://ivul.kaust.edu.sa/benchmark-and-simulator-uav-tracking-dataset), Matthias Mueller, Neil Smith, Bernard Ghanem, ECCV 2016.

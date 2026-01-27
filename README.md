# Formel 1 Vorhersage Projekt

Dieses Projekt wurde im Rahmen einer Projekt erstellt und beschäftigt sich mit der Vorhersage von verschiedenen Ereignissen in der Formel 1.

## Projektstruktur

Das Projekt ist in die folgenden Ordner unterteilt:

- `data`: Enthält alle Skripte und Dateien zur Erstellung und Verarbeitung der Daten. Die Rohdaten sowie die verarbeiteten Datensätze sind hier zu finden.
- `notebooks`: Enthält alle Jupyter Notebooks zur Analyse und zum Training der Modelle.
- `scripts`: Enthält verschiedene Python-Skripte zur Automatisierung von Aufgaben wie Datenauswertung und Modelltraining.
- `ui`: React-Frontend (Vite + MUI) für die Inferenz der Punkte-Klassifikation.
- `thesis`: Enthält die LaTeX-Dateien der Projektarbeit.

## Abhängigkeiten

Das Projekt hat die folgenden Abhängigkeiten:

- `matplotlib`
- `fastf1`
- `seaborn`
- `tqdm`
- `scikit-learn`
- `torch`
- `tabpfn`
- `numpy`
- `pandas`
- `lightgbm`
- `fastapi`
- `uvicorn`

Die Abhängigkeiten können mit dem folgenden Befehl installiert werden:

```bash
pip install -r requirements.txt
```

Stellen Sie sicher, dass Sie die virtuelle Umgebung aktiviert haben, bevor Sie die Abhängigkeiten installieren.

## Ziel des Projekts

Das Ziel dieses Projekts ist die Vorhersage von verschiedenen Ereignissen in der Formel 1. Dazu werden verschiedene maschinelle Lernmodelle trainiert und evaluiert. Die Ergebnisse werden in der Projektarbeit "Vorhersagen verschiedener Formel 1 Events" von Fiete Scheel dokumentiert.

## Skripte

Die Skripte sind in zwei Hauptordner unterteilt: `data/scripts` für die Datenverarbeitung und `scripts` für die Analyse.

### `data/scripts`

Diese Skripte sind für die Erstellung, Aufbereitung und Aufteilung der Datensätze verantwortlich.

- `build_fastf1_dataset_points_scored.py`: Erstellt einen Datensatz für die Klassifikationsaufgabe, ob ein Fahrer Punkte erzielt. Das Skript lädt Daten von FastF1, extrahiert Features wie Grid-Position, Qualifying-Zeiten, WM-Stände und berechnet am Ende das Label `points_scored`.
  - **Ausführung:**
    ```bash
    python data/scripts/build_fastf1_dataset_points_scored.py --years 2022 2023 --output data/points_scored/grandprix_features.csv
    ```
- `build_fastf1_dataset_regression.py`: Erstellt einen Datensatz für die Regressionsaufgabe (z.B. Vorhersage der Rennzeit). Ähnlich wie das Klassifikations-Skript, aber mit dem Ziel, die Rennzeit oder den Abstand zum Sieger (`gap_to_winner`) zu extrahieren. Es können auch Daten aus den freien Trainings (`FP1`, `FP2`, `FP3`) mit einbezogen werden.
  - **Ausführung:**
    ```bash
    python data/scripts/build_fastf1_dataset_regression.py --years 2022 2023 --output data/regression/grandprix_features.csv
    ```
- `merge_csvs.py`: Fügt mehrere CSV-Dateien aus einem Verzeichnis zu einer einzigen Datei zusammen.
  - **Ausführung:**
    ```bash
    python data/scripts/merge_csvs.py data/regression/years --output data/regression/grandprix_features_all.csv
    ```
- `split_dataset.py`: Teilt eine einzelne CSV-Datei auf Basis von Jahren in Trainings-, Validierungs- und Test-Sets auf, um die zeitliche Abhängigkeit zu wahren.
  - **Ausführung:**
    ```bash
    python data/scripts/split_dataset.py --data data/points_scored/grandprix_features_all.csv --year-column year --test-size 0.2 --val-size 0.25
    ```
- `split_years_dataset.py`: Teilt Datensätze, die bereits pro Saison in einzelnen CSVs vorliegen, in Trainings-, Validierungs- und Test-Sets auf, indem die Jahre explizit zugewiesen werden.
  - **Ausführung:**
    ```bash
    python data/scripts/split_years_dataset.py --input-dir data/points_scored/years --val-years 2022 --test-years 2023 --output-prefix data/points_scored/dataset_years
    ```

### `scripts`

Diese Skripte dienen der Analyse und Auswertung der erstellten Datensätze.

- `avg_driver_distance.py`: Berechnet die durchschnittliche absolute Differenz zwischen den Fahrern pro Rennen für eine bestimmte Zielspalte (z.B. `gap_to_winner`).
  - **Ausführung:**
    ```bash
    python scripts/avg_driver_distance.py --csv data/regression/grandprix_features_all.csv --target gap_to_winner
    ```
- `calc_retirement_rate.py`: Berechnet die durchschnittliche Ausfallquote über bestimmte Formel-1-Saisons hinweg mithilfe von FastF1.
  - **Ausführung:**
    ```bash
    python scripts/calc_retirement_rate.py --years 2022 2023
    ```
- `run_learning_curve.py`: Führt eine Lernkurvenanalyse durch, indem die Anzahl der Trainingssaisons erhöht wird. Es trainiert ein LightGBM-Modell und gibt die Metriken aus.
  - **Ausführung (Beispiel):**
    ```bash
    python scripts/run_learning_curve.py \
        --data-dir data/points_scored/years \
        --train-sets "2018,2019" "2018,2019,2020" "2018,2019,2020,2021" \
        --val-years 2022 \
        --test-years 2023 \
        --output learning_curve_by_year.csv
    ```
- `run_learning_curve_sizes.py`: Führt eine Lernkurvenanalyse durch, indem die Größe des Trainingsdatensatzes variiert wird. Es werden Trainings-Subsets erstellt und ein LightGBM-Modell trainiert.
  - **Ausführung (Beispiel):**
    ```bash
    python scripts/run_learning_curve_sizes.py \
      --train data/points_scored/grandprix_features_train.csv \
      --val data/points_scored/grandprix_features_val.csv \
      --test data/points_scored/grandprix_features_test.csv \
      --train-sizes 500 1000 2000 3000 \
      --output data/learning_curve_sizes.csv \
      --subset-dir data/train_subsets
    ```

### Inferenz (TabPFN) + API

Für die Klassifikationsaufgabe `points_scored` gibt es ein Inferenz-Setup mit
gespeichertem TabPFN-Bundle und einer kleinen FastAPI:

- `scripts/points_scored_model.py`: Gemeinsame Feature-Logik (CAT/NUM, Encoding).
- `scripts/train_tabpfn_points_scored.py`: Fit + Speichern des TabPFN-Bundles.
- `scripts/predict_points_scored.py`: CLI-Inferenz via JSON-Input.
- `scripts/serve_points_scored.py`: FastAPI-Server mit `/predict`.

**Workflow:**
```bash
# 1) Modell fitten und speichern
python scripts/train_tabpfn_points_scored.py

# 2) API starten
python -m uvicorn scripts.serve_points_scored:app --reload
```

**Beispiel-Request (CLI):**
```bash
python scripts/predict_points_scored.py --input '{"driver_id":"hamilton","constructor_id":"mercedes","circuit_id":"melbourne","year":2015,"grid_position":1,"quali_delta":0.0,"quali_tm_delta":-0.59,"season_pts_driver":0.0,"season_pts_team":0.0,"last_3_avg":0.0,"is_street_circuit":1,"is_wet":0}'
```

### UI (React + MUI)

Das Frontend erlaubt das manuelle Eingeben der Features und zeigt Prediction
plus Wahrscheinlichkeit an. Es nutzt Vite und Material UI.

```bash
cd ui
npm install
npm run dev
```

## Notebooks

Der Ordner `notebooks` ist in zwei Hauptkategorien unterteilt: `points_scored` und `regression`.

### `points_scored`

Diese Notebooks konzentrieren sich auf die Klassifikationsaufgabe, ob ein Fahrer Punkte erzielt oder nicht.

- `baseline_training.ipynb`: Training eines einfachen Baseline-Modells.
- `learning_curve_analysis.ipynb`: Analyse der mit den Skripten `run_learning_curve.py` und `run_learning_curve_sizes.py` erstellten Lernkurven.
- `model_training.ipynb`: Training verschiedener Modelle für die Punkte-Klassifikation.
- `random_forest_baseline.ipynb`: Training eines Random-Forest-Modells als Baseline.
- `tabpfn.ipynb`: Experimente mit dem TabPFN-Modell.
- `xgboost_training.ipynb`: Training eines XGBoost-Modells.

### `regression`

Diese Notebooks konzentrieren sich auf die Regressionsaufgabe, den Abstand zum Sieger (`gap_to_winner`) vorherzusagen.

- `baseline_regression.ipynb`: Training eines einfachen Baseline-Regressionsmodells.
- `gap_to_winner_regression.ipynb`: Detaillierte Analyse und Modellierung für die Vorhersage des `gap_to_winner`.
- `random_forest_regression.ipynb`: Training eines Random-Forest-Regressionsmodells.
- `tabpfn_regression.ipynb`: Experimente mit dem TabPFN-Modell für die Regression.
- `xgboost_regression.ipynb`: Training eines XGBoost-Regressionsmodells.

#### `regression/with_training_sessions`

Dieser Unterordner enthält Notebooks, die zusätzlich Daten aus den Trainings-Sessions für die Regressionsaufgabe verwenden.

- `baseline_regression.ipynb`: Baseline-Modell mit Trainingsdaten.
- `gap_to_winner_regression.ipynb`: Regressionsanalyse mit Trainingsdaten.
- `random_forest_regression.ipynb`: Random-Forest-Modell mit Trainingsdaten.
- `tabpfn_regression.ipynb`: TabPFN-Modell mit Trainingsdaten.
- `xgboost_regression.ipynb`: XGBoost-Modell mit Trainingsdaten.

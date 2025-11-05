# 3_validation_docking/main.py
import sys
import os
import logging

# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import config

from src.data_preparer import DataPreparer
from src.ligand_preparer import generate_3d_conformers
from src.docking_runner import DockingRunner
from src.results_analyzer import ResultsAnalyzer

def main():
    """
    Полный пайплайн: подготовка данных, докинг и анализ результатов.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("--- ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ВАЛИДАЦИИ (ДОКИНГ И АНАЛИЗ) ---")
    
    docking_config = config.DOCKING_CONFIG

    # --- Шаг 1: Объединение и дедупликация входных данных ---
    data_prep = DataPreparer(docking_config)
    combined_csv_path = data_prep.combine_and_deduplicate()
    if not combined_csv_path: return

    # --- Шаг 2: Подготовка 3D структур лигандов ---
    generate_3d_conformers(
        csv_path=combined_csv_path, 
        output_dir=docking_config["PREPARED_LIGANDS_DIR"]
    )

    # --- Шаг 3: Подготовка и запуск докинга ---
    runner = DockingRunner(docking_config)
    runner.prepare_receptor()
    box_params = runner.calculate_docking_box()
    runner.run_docking_batch(box_params)

    # --- Шаг 4: Анализ результатов и построение графиков ---
    analyzer = ResultsAnalyzer(docking_config)
    final_csv_path = analyzer.parse_docking_logs(combined_csv_path)
    if final_csv_path:
        analyzer.create_analysis_plots(final_csv_path)

    logging.info("Пайплайн валидации успешно завершен.")

if __name__ == "__main__":
    main()
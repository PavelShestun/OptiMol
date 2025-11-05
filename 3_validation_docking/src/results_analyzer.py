# 3_validation_docking/src/results_analyzer.py
import os
import re
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

class ResultsAnalyzer:
    def __init__(self, config: dict):
        self.config = config

    def parse_docking_logs(self, combined_csv_path: str) -> str:
        """
        Парсит логи GNINA, извлекает аффинность и добавляет ее в CSV файл.
        Возвращает путь к финальному CSV с результатами докинга.
        """
        logging.info("Начало анализа логов докинга...")
        try:
            df = pd.read_csv(combined_csv_path)
        except FileNotFoundError:
            logging.error(f"Файл не найден: {combined_csv_path}")
            return None

        affinities = []
        for i in range(len(df)):
            ligand_name = f"ligand_{i + 1}"
            log_file = os.path.join(self.config["DOCKING_RESULTS_DIR"], f"{ligand_name}_log.txt")
            if os.path.exists(log_file):
                affinities.append(self._extract_best_affinity(log_file))
            else:
                affinities.append(None)
        
        df['Docking_Affinity'] = affinities
        
        final_csv_path = os.path.join(
            os.path.dirname(self.config["DOCKING_RESULTS_DIR"]), 
            "final_results_with_docking.csv"
        )
        df.to_csv(final_csv_path, index=False)
        logging.info(f"Результаты докинга добавлены. Финальный датасет сохранен в: {final_csv_path}")
        return final_csv_path

    def _extract_best_affinity(self, log_file: str) -> float or None:
        """Извлекает лучшую (минимальную) аффинность из лог-файла GNINA."""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                # Ищем все значения аффинности в строках, начинающихся с "CNNscore" или "CNNaffinity"
                affinity_values = re.findall(r'Affinity:\s*(-?\d+\.\d+)', content)
                if affinity_values:
                    return min(float(a) for a in affinity_values)
        except Exception as e:
            logging.warning(f"Не удалось прочитать аффинность из {log_file}: {e}")
        return None
    
    def create_analysis_plots(self, final_csv_path: str):
        """Создает и сохраняет набор аналитических графиков."""
        logging.info("Создание аналитических графиков...")
        try:
            df = pd.read_csv(final_csv_path).dropna(subset=['Docking_Affinity'])
        except (FileNotFoundError, pd.errors.EmptyDataError):
            logging.error("Финальный CSV с результатами докинга не найден или пуст.")
            return

        os.makedirs(self.config["ANALYSIS_RESULTS_DIR"], exist_ok=True)
        sns.set_style("whitegrid")

        # 1. Boxplot распределения аффинности
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Model', y='Docking_Affinity', data=df)
        plt.title('Распределение Аффинности Докинга по Моделям')
        plt.xlabel('Модель')
        plt.ylabel('Аффинность (kcal/mol)')
        plt.savefig(os.path.join(self.config["ANALYSIS_RESULTS_DIR"], "1_affinity_distribution.png"))
        plt.close()

        # 2. Визуализация топ-10 молекул
        top_df = df.sort_values(by='Docking_Affinity', ascending=True).head(10)
        mols = [Chem.MolFromSmiles(smi) for smi in top_df['SMILES']]
        legends = [f"Aff: {aff:.2f} kcal/mol\nModel: {model}" 
                   for aff, model in zip(top_df['Docking_Affinity'], top_df['Model'])]
        try:
            img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(250, 250), legends=legends)
            img.save(os.path.join(self.config["ANALYSIS_RESULTS_DIR"], "2_top10_molecules.png"))
        except Exception as e:
            logging.warning(f"Could not generate molecule images: {e}")
        
        logging.info(f"Аналитические графики сохранены в: {self.config['ANALYSIS_RESULTS_DIR']}")

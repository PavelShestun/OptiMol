import os
import requests
import logging
import stat
import subprocess
from tqdm import tqdm
import pandas as pd

class DockingRunner:
    """
    Класс для управления процессом молекулярного докинга с GNINA.
    """
    def __init__(self, config: dict):
        self.config = config
        self._check_and_download_gnina()

    def _check_and_download_gnina(self):
        """Проверяет наличие GNINA и скачивает, если необходимо."""
        path = self.config["GNINA_EXECUTABLE_PATH"]
        if os.path.exists(path):
            logging.info(f"Исполняемый файл GNINA найден: {path}")
            return

        logging.info("GNINA не найдена. Попытка скачивания...")
        url = "https://github.com/gnina/gnina/releases/download/v1.3/gnina"
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(path, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Скачивание GNINA"):
                    f.write(chunk)
            # Делаем файл исполняемым
            st = os.stat(path)
            os.chmod(path, st.st_mode | stat.S_IEXEC)
            logging.info("GNINA успешно скачана и готова к работе.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка скачивания GNINA: {e}")
            raise

    def prepare_receptor(self):
        """Скачивает PDB файл рецептора и очищает его от не-белковых атомов."""
        os.makedirs(self.config["RECEPTOR_DIR"], exist_ok=True)
        
        full_pdb_path = os.path.join(self.config["RECEPTOR_DIR"], f"{self.config['PDB_ID']}_full.pdb")
        
        if not os.path.exists(full_pdb_path):
            logging.info(f"Скачивание PDB файла {self.config['PDB_ID']}...")
            response = requests.get(f"https://files.rcsb.org/download/{self.config['PDB_ID']}.pdb")
            response.raise_for_status()
            with open(full_pdb_path, "wb") as f:
                f.write(response.content)
            logging.info(f"Полный PDB файл успешно скачан в {full_pdb_path}.")

        prepared_receptor_path = os.path.join(self.config["RECEPTOR_DIR"], f"{self.config['PDB_ID']}_receptor.pdb")
        logging.info("Подготовка файла рецептора (удаление не-белковых атомов)...")
        with open(full_pdb_path, 'r') as infile, open(prepared_receptor_path, 'w') as outfile:
            for line in infile:
                if line.startswith('ATOM'):
                    outfile.write(line)
        logging.info(f"Файл рецептора готов: {prepared_receptor_path}")

    def calculate_docking_box(self):
        """Рассчитывает координаты докинг-бокса по референсному лиганду."""
        logging.info("Расчет координат докинг-бокса...")
        coords = []
        full_pdb_path = os.path.join(self.config["RECEPTOR_DIR"], f"{self.config['PDB_ID']}_full.pdb")

        with open(full_pdb_path, 'r') as f:
            for line in f:
                if line.startswith('HETATM') and \
                   line[17:20].strip() == self.config["REF_LIGAND_RES_NAME"] and \
                   line[21:22].strip() == self.config["REF_LIGAND_CHAIN"] and \
                   line[22:26].strip() == self.config["REF_LIGAND_RES_NUM"]:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

        try:
            logging.info("Attempting to predict docking box with P2Rank...")
            prepared_receptor_path = os.path.join(self.config["RECEPTOR_DIR"], f"{self.config['PDB_ID']}_receptor.pdb")
            box_params = self._get_p2rank_prediction(prepared_receptor_path)
            logging.info(f"P2Rank predicted docking box: { {k: f'{v:.3f}' for k, v in box_params.items()} }")
            return box_params
        except Exception as e:
            logging.warning(f"P2Rank prediction failed: {e}. Falling back to reference ligand method.")
            if not coords:
                raise ValueError("Референсный лиганд для определения докинг-бокса не найден в PDB файле.")

            coords = pd.DataFrame(coords, columns=['x', 'y', 'z'])
            center = coords.mean()
            size = (coords.max() - coords.min()) + self.config["BOX_PADDING"]

            box_params = {
                'center_x': center['x'], 'center_y': center['y'], 'center_z': center['z'],
                'size_x': size['x'], 'size_y': size['y'], 'size_z': size['z']
            }
            logging.info(f"Координаты докинг-бокса рассчитаны: { {k: f'{v:.3f}' for k, v in box_params.items()} }")
            return box_params

    def _get_p2rank_prediction(self, pdb_file_path):
        """Submits a PDB file to PrankWeb and retrieves the top pocket."""
        import time

        # Create configuration file
        config_path = "p2rank_config.json"
        with open(config_path, "w") as f:
            f.write('{"chains":[], "structure-sealed": true, "prediction-model": "default"}')

        # Submit to PrankWeb
        with open(pdb_file_path, 'rb') as pdb_file:
            files = {
                'structure': (os.path.basename(pdb_file_path), pdb_file, 'chemical/x-pdb'),
                'configuration': (config_path, open(config_path, 'rb'), 'application/json')
            }
            response = requests.post("https://prankweb.cz/api/v2/prediction/v4-user-upload", files=files)
            response.raise_for_status()

        prediction_data = response.json()
        task_id = prediction_data.get("id")

        # Poll for results
        status_url = f"https://prankweb.cz/api/v2/prediction/v4-user-upload/{task_id}"
        while True:
            status_response = requests.get(status_url)
            status_data = status_response.json()
            if status_data.get("status") == "successful":
                break
            time.sleep(10)

        # Download and parse results
        pockets_url = f"https://prankweb.cz/api/v2/prediction/v4-user-upload/{task_id}/pockets"
        pockets_response = requests.get(pockets_url)
        pockets_data = pockets_response.json()

        if not pockets_data:
            raise ValueError("P2Rank did not return any pockets.")

        top_pocket = pockets_data[0]
        center = top_pocket['center']
        size = top_pocket.get('size', {'x': 20, 'y': 20, 'z': 20}) # Default size if not provided

        return {
            'center_x': center['x'], 'center_y': center['y'], 'center_z': center['z'],
            'size_x': size['x'], 'size_y': size['y'], 'size_z': size['z']
        }

    def run_docking_batch(self, box_params: dict):
        """Запускает докинг для всех подготовленных лигандов."""
        logging.info("Начало процесса докинга...")
        os.makedirs(self.config["DOCKING_RESULTS_DIR"], exist_ok=True)
        
        ligand_files = [f for f in os.listdir(self.config["PREPARED_LIGANDS_DIR"]) if f.endswith(".pdb")]
        if not ligand_files:
            logging.error(f"В папке {self.config['PREPARED_LIGANDS_DIR']} не найдено подготовленных лигандов (.pdb).")
            return

        prepared_receptor_path = os.path.join(self.config["RECEPTOR_DIR"], f"{self.config['PDB_ID']}_receptor.pdb")

        for ligand_file in tqdm(ligand_files, desc="Докинг лигандов"):
            ligand_name = os.path.splitext(ligand_file)[0]
            ligand_path = os.path.join(self.config["PREPARED_LIGANDS_DIR"], ligand_file)
            
            output_sdf = os.path.join(self.config["DOCKING_RESULTS_DIR"], f"docked_{ligand_name}.sdf")
            log_file = os.path.join(self.config["DOCKING_RESULTS_DIR"], f"{ligand_name}_log.txt")

            command = [
                self.config["GNINA_EXECUTABLE_PATH"],
                '-r', prepared_receptor_path,
                '-l', ligand_path,
                '--center_x', str(box_params['center_x']),
                '--center_y', str(box_params['center_y']),
                '--center_z', str(box_params['center_z']),
                '--size_x', str(box_params['size_x']),
                '--size_y', str(box_params['size_y']),
                '--size_z', str(box_params['size_z']),
                '-o', output_sdf,
                '--seed', str(self.config["DOCKING_SEED"]),
                '--cnn', self.config["CNN_MODE"]
            ]

            try:
                with open(log_file, "w") as f_log:
                    subprocess.run(command, check=True, stdout=f_log, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                logging.error(f"Ошибка при докинге {ligand_file}. Подробности в {log_file}")
            except Exception as e:
                logging.error(f"Неожиданная ошибка при докинге {ligand_file}: {e}")

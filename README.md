# Comprehensive Pipeline for De Novo Design of KEAP1 Inhibitors

<img width="871" height="484" alt="Frame 1 (1)" src="https://github.com/user-attachments/assets/ce42b775-15b1-4f73-a64f-b40afd8716ca" />

This repository contains a complete set of tools for generating, evaluating, and validating new molecular inhibitors of the KEAP1 protein. The project is divided into three independent but interconnected applications designed to address challenges within the framework of Alzheimer's disease therapy.

## 🚀 Features

-   **SELFIES-based Generation**: Utilizes the SELFIES (SELF-referencIng Embedded Strings) representation to ensure 100% validity of generated molecular structures.
-   **Reinforcement Learning (RL)**: An RNN is trained using RL to optimize a multi-objective reward function.
-   **Multi-Objective Optimization**: The reward function includes predicted activity (pIC50), selectivity, QED, CNS MPO, BBB score, and Synthetic Accessibility (SA Score).
-   **Predictive Models**: Ensemble models (Stacking Regressor) trained on ChEMBL data are used to predict activity.
-   **Pareto Filtering**: Used to select the best trade-off solutions from the generated molecules.
-   **Structural Validation**: Final candidate selection is confirmed via molecular docking using GNINA.
-   **Modular Structure**: Code is organized into modules to improve readability and reusability.

## 🎯 Target Selection Rationale

Our choice of target, **KEAP1 (Kelch-like ECH-associated protein 1)**, is based on a comprehensive analysis of current scientific literature, which indicated the need to shift focus from "classical" hypotheses to more fundamental and interconnected pathological processes.

### 1. Beyond the Amyloid Hypothesis
Clinical trials of drugs targeting amyloid-beta (Aβ) and tau proteins have often demonstrated low efficacy. This has led the scientific community to rethink therapeutic strategies and search for alternative, more promising targets underlying the disease.

### 2. KEAP1 as a Multi-Target Regulator
KEAP1 allows for the implementation of a **multi-target strategy** through the modulation of just a single protein. Inhibition of KEAP1 leads to the activation of the transcription factor **Nrf2**, which simultaneously:
*   **Reduces neuroinflammation** by suppressing the expression of pro-inflammatory cytokines.
*   **Combats oxidative stress** acting as the "master regulator" of the cellular antioxidant response.

This dual mechanism of action makes KEAP1 a relevant and scientifically grounded target.

### 3. Data Availability

*   **Activity Data Availability:** A ready-made dataset exists in ChEMBL (`CHEMBL2069156`) for IC50 with examples of highly active ligands. This is the only dataset meeting the requirements (Organism: Human, Type: Single Protein). This dataset is used to train the predictor.
*   **Structural Data Availability:** The crystal structure of the KEAP1 complex is available in the PDB (`ID: 4L7B`) with a resolution of 2.41 Å for molecular docking.

---
The full analysis leading to this choice is detailed in our literature review:
[https://docs.google.com/document/d/1kjdbpZcpU789loig5updVyEvMRaqEIOzgz5KDh-P73c/edit?usp=sharing](https://docs.google.com/document/d/1kjdbpZcpU789loig5updVyEvMRaqEIOzgz5KDh-P73c/edit?usp=sharing)

## 🏗️ Model Architecture Rationale

Our project is built on the **interaction of two fundamentally different but complementary architectures**, each chosen to solve its specific task: one for **prediction (evaluation)** and the other for **generation (creation)**.

### 1. Predictive Model Architecture
**Task:** Create a maximally accurate and reliable function that can predict biological activity (pIC50) based on the molecular structure. This model will guide our generator during the reinforcement learning process.

**Chosen Architecture and Rationale:**
*   **Feature Representation:** We do not feed SMILES directly. The molecule is first converted into a numerical vector consisting of:
    1.  **Morgan Fingerprints:** Encode the presence of specific cyclic substructures in the molecule.
    2.  **Physico-chemical Descriptors:** We add key properties (LogP, TPSA, weight, etc.) that are important for the molecule's behavior in the organism.
*   **Model Ensemble and Stacking:** We organized a "competition" between algorithms (`RandomForest`, `XGBoost`, `LightGBM`) and combined them using a **`StackingRegressor`**. This ensembling method creates a meta-model that learns from the predictions of the base models, resulting in a final predictor that is more accurate and stable than any single model alone.

### 2. Generative Model Architecture
**Task:** Create a model capable of generating new, chemically correct, diverse, and highly active molecules by learning from feedback provided by the predictor.

**Chosen Architecture and Rationale:**
*   **Molecular Representation (SELFIES):** We used **SELFIES**—a string representation whose grammar **guarantees 100% validity** for any generated string.
*   **Generator Core (LSTM-based RNN):** To generate SELFIES sequences, we chose a Recurrent Neural Network (RNN) with LSTM cells. Its architecture consists of three key parts:
    1.  **Embedding Layer:** Converts each SELFIES token into a numerical vector, allowing the network to understand chemical "semantics".
    2.  **Stack of 3 LSTM Layers:** The core of the network. LSTMs are ideal for working with sequences because their internal "gate" structure allows them to "remember" information over long distances (e.g., remembering to close a ring opened at the beginning of the molecule). Multiple layers allow for learning hierarchical, more complex patterns.
    3.  **Fully Connected Layer (Output):** Converts the LSTM output into a probability distribution over the entire vocabulary, from which the next token for generation is selected.
*   **Training Strategy (Pre-training + RL):**
    1.  **Pre-training on MOSES:** First, we train the generator on a large dataset so it learns the basic "grammar" of chemistry.
    2.  **Fine-tuning via RL:** The trained model is then fine-tuned to solve the specific task using scores from our predictor as rewards.

## 📈 Multi-Criteria Optimization and Key Metrics

Our pipeline does not just generate molecules but purposely optimizes them across a wide range of criteria to obtain candidates with the best balance of properties. The evaluation process can be divided into several key areas:

### 1. Efficacy and Selectivity
These are the primary biological goals of our project.

*   **Predicted Activity (pIC50_KEAP1):** The main target metric. This is the negative decimal logarithm of the half-maximal inhibitory concentration (IC50) for our KEAP1 target. **The higher this value, the more potently the molecule inhibits the target.** We aim to maximize this indicator.
*   **Selectivity Score:** Shows how selectively the molecule acts on KEAP1 compared to two anti-targets (EGFR and IKKb). Calculated as `pIC50_KEAP1 - max(pIC50_EGFR, pIC50_IKKb)`. **A high positive value indicates high selectivity**, which is critically important for reducing side effects.

### 2. Drug-Likeness Properties
These metrics evaluate how much the molecule resembles a successful drug overall.

*   **Quantitative Estimate of Drug-likeness (QED):** An aggregated metric (from 0 to 1) based on the distribution of key physico-chemical properties in already approved drugs. **A value close to 1 indicates a high "drug-likeness" profile.**
*   **Synthetic Accessibility (SA Score):** An assessment of the complexity of synthesizing the molecule (from 1 to 10). Based on fragment analysis and structural complexity. **The lower the value, the easier (theoretically) it is to synthesize the molecule.**
*   **Lipinski's Rule of Five:** A set of empirical rules to evaluate the probability that a molecule will have good oral bioavailability. We filter molecules that do not meet the following criteria:
    *   Molecular Weight (MolWt) ≤ 500
    *   Partition Coefficient (LogP) ≤ 5
    *   Number of Hydrogen Bond Donors (HBD) ≤ 5
    *   Number of Hydrogen Bond Acceptors (HBA) ≤ 10
*   **Veber's Rule:** Additional criteria for good bioavailability.
    *   Topological Polar Surface Area (TPSA) ≤ 140 Å²
    *   Number of Rotatable Bonds (NumRotatableBonds) ≤ 10

### 3. CNS Penetration Properties
Since Alzheimer's is a disease of the Central Nervous System (CNS), these properties are critically important.

*   **Blood-Brain Barrier (BBB) Score:** Our custom score (from 0 to 1) that penalizes molecules for deviating from optimal values for BBB penetration regarding LogP, TPSA, molecular weight, and hydrogen bond donors/acceptors.
*   **CNS Multi-Parameter Optimization (CNS MPO):** A more complex aggregated metric (from 0 to 6) that evaluates the suitability of a molecule for CNS targeting.

### 4. Safety and Structural Filters
These filters help screen out undesirable or potentially toxic structures at an early stage.

*   **PAINS Filter (Pan-Assay Interference Compounds):** Excludes molecules containing substructures known to non-specifically interact with proteins and yield false-positive results in biological assays.
*   **Ring Count:** We introduced a requirement in the reward function for the presence of at least one ring in the structure to filter out primitive linear molecules and guide generation toward more complex, drug-like scaffolds.

### 5. Generation Quality Metrics
These metrics evaluate the generation process itself rather than a specific molecule, allowing us to understand how well the model is performing.

*   **Diversity:** Evaluates how different the molecules generated in one batch are from each other. High diversity suggests the model is not "stuck" on a single idea.
*   **Novelty:** Shows how different the generated molecules are from those the model was trained on. High novelty is a sign that the model is creating truly new structures.
*   **Uniqueness:** The percentage of unique molecules in a generated batch. A value of 100% means the model does not produce duplicates.

## 🛠️ Workflow and Pipeline Evolution

Development proceeded iteratively, where each stage solved the problems of the previous one.

**1. Start and Technical Obstacles.**
*   **Problem:** The initial ambitious pipeline, which included molecular docking, was non-functional due to critical errors: dependency conflicts for docking and a tensor dimension error in the LSTM generator.
*   **Solution:** We made a strategic decision to completely remove docking from the main RL loop (leaving it for final validation) and fixed the bug in the generator code.

**2. Conceptual Issue: "Reward Hacking"**
*   **Problem:** After fixing the errors, the model "learned to cheat" the simple reward function by generating chemically primitive but formally "profitable" linear molecules with zero medicinal value.
*   **Solution:** We complicated the reward function by adding an explicit incentive for creating ring structures typical of drugs.

**3. Conceptual Issue: "Mode Collapse"**
*   **Problem:** The model found one successful chemical scaffold and began to "exploit" it, leading to a loss of diversity—all the best molecules began to look alike.
*   **Solution:** To encourage "exploratory curiosity," we introduced a **novelty bonus** to the reward function, calculated based on the dissimilarity (using Tanimoto metric) of the new molecule to those already generated.

## 🔬 Comparative Experiments and Validation

### 1. Parallel Experiment with DiffSBDD
We conducted an experiment using the ready-made tool **DiffSBDD** for generating molecules within the KEAP1 active site. DiffSBDD is an SE(3)-equivariant diffusion model that generates new ligands taking into account protein pockets. This model was chosen as one of the cutting-edge tools in its field. By applying our evaluation system to its results, we confirmed that our custom RL pipeline provides **better control over multi-parameter optimization** (activity, selectivity, ADMET), which is critically important for complex tasks.

### 2. Final Validation: Molecular Docking with GNINA
As a concluding stage, we performed molecular docking for the best candidates selected by our pipeline.
*   **Tool:** We used **GNINA**, which applies CNNs to improve the scoring function and works with 3D structures.
*   **Process:** The best molecules were docked to the KEAP1 active site (`PDB ID: 4L7B`).
*   **Result:** Docking served as a **final independent filter**, confirming that our candidates not only possess good calculated properties but are also capable of high-affinity interaction within the target binding pocket.

## 📁 Project Structure
```
DataCon2025_DrugDiscovery/
│
├── 1_generative_rl_model/      # << Application 1: Your RL Model
│   ├── models/
│   ├── src/
│   └── main.py
│
├── 2_comparison_diffsbdd/      # << Application 2: Comparison with DiffSBDD
│   └── main.py
│
├── 3_validation_docking/       # << Application 3: Docking with GNINA
│   ├── src/
│   └── main.py
│
├── config.py                   # << SHARED configuration file for all
├── requirements.txt
└── README.md
```

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PavelShestun/DataCon2025.git
    cd DataCon2025_DrugDiscovery
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/macOS
    # venv\Scripts\activate    # For Windows
    ```

3.  **Install dependencies:**
    ```bash
    # Install RDKit via Conda first (recommended)
    conda install -c conda-forge rdkit
    # Then install the remaining packages
    pip install -r requirements.txt
    ```

## ▶️ How to Use: Step-by-Step Pipeline

The project is designed for sequential execution.

### Step 1: Molecule Generation (Your RL Model)
This stage creates new molecules and, importantly, trains the predictor models needed for the subsequent steps.

1.  **Navigate to the application directory:**
    ```bash
    cd 1_generative_rl_model
    ```
2.  **Run the pipeline:**
    ```bash
    python main.py
    ```
3.  **Result:** The file `final_molecules.csv` will appear in the `results/1_rl_model_outputs/` folder. The trained predictors will be saved in `1_generative_rl_model/models/`.

### Step 2: Comparative Generation (DiffSBDD)
This stage launches the alternative generator and evaluates its results.

1.  **Return to root and navigate to the application directory:**
    ```bash
    cd ../2_comparison_diffsbdd
    ```
2.  **Run the pipeline:**
    ```bash
    python main.py
    ```
3.  **Result:** The file `diffsbdd_evaluated.csv` will appear in the `results/2_diffsbdd_outputs/` folder.

### Step 3: Final Validation (Docking)
This stage performs structural validation for any set of candidates.

1.  **Configure the input file:** Open the root `config.py` and in the `DOCKING_CONFIG` section, specify which CSV file you want to dock:
    ```python
    # in config.py
    DOCKING_CONFIG = {
        # Dock the result of the RL model
        "INPUT_CSV": RL_CONFIG["OUTPUT_CSV"], 
        # Or uncomment the line below to dock the DiffSBDD result
        # "INPUT_CSV": DIFFSBDD_CONFIG["OUTPUT_CSV"], 
        ...
    }
    ```
2.  **Return to root and navigate to the application directory:**
    ```bash
    cd ../3_validation_docking
    ```
3.  **Run the pipeline:**
    ```bash
    python main.py
    ```
4.  **Result:** Docking results will appear in the `results/3_docking_outputs/` folder: SDF files with poses and text logs for each molecule.

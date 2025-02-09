# **ReutersNOZ - Setup Guide**  

## **1. Clone the Repository**  
Clone the repository to your local machine using the following command:  

```bash
git clone git@github.com:RiccardoDAndrea/ReutersNOZ.git
cd ReutersNOZ
```

## **2. Set Up a Virtual Environment**  
Before installing dependencies, create and activate a virtual environment:

```bash
python -m venv venv
```

### **Activate the Virtual Environment**  
- **Linux/Mac:**  
  ```bash
  source venv/bin/activate
  ```
- **Windows (PowerShell):**  
  ```powershell
  venv\Scripts\Activate
  ```

## **3. Install Dependencies**  
Once the virtual environment is activated, install the required dependencies:  

```bash
pip install -r requirements.txt
```

## **4. Run the Main Script**  
Execute `main.py`. The script may take a few seconds to complete:  

```bash
python main.py
```

## **5. Perform Data Analysis**  
For further analysis, open the `quickanalyse.ipynb` notebook using **Jupyter Notebook**:  

```bash
jupyter notebook Analyse/quickanalyse.ipynb
```

Running the notebook will generate various visualizations, including a **word cloud** of the most frequently used topics.  
Additionally, the word cloud is saved as **`topic_freq.png`**, providing an overview of the topic distribution.

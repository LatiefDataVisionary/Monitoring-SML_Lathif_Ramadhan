import os
import json
import logging
import requests
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

MODEL_URL = "http://localhost:5000/invocations"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "test_cleaned.csv")

class InferenceClient:
    """
    Kelas klien untuk melakukan pengujian inferensi ke endpoint model MLflow.
    """
    def __init__(self, endpoint_url: str, data_path: str):
        self.endpoint_url = endpoint_url
        self.data_path = data_path
        self.headers = {"Content-Type": "application/json"}

    def load_sample_data(self, sample_size: int = 1) -> dict:
        """
        Memuat data uji dan mengubahnya ke dalam format yang diterima oleh MLflow (dataframe_split).
        """
        try:
            df = pd.read_csv(self.data_path)
            if 'Churn' in df.columns:
                df = df.drop(columns=['Churn'])
            
            sample_df = df.sample(n=sample_size, random_state=None)
            
            payload = {
                "dataframe_split": {
                    "columns": sample_df.columns.tolist(),
                    "data": sample_df.values.tolist()
                }
            }
            return payload
        except Exception as e:
            logging.error(f"Gagal memuat data: {str(e)}")
            raise

    def predict(self, payload: dict):
        """
        Mengirimkan payload ke endpoint model dan mengembalikan hasil prediksi.
        """
        try:
            logging.info(f"Mengirim request ke {self.endpoint_url}...")
            response = requests.post(self.endpoint_url, data=json.dumps(payload), headers=self.headers)
            
            if response.status_code == 200:
                predictions = response.json()
                logging.info(f"Prediksi berhasil diterima: {predictions}")
                return predictions
            else:
                logging.error(f"Request gagal dengan status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Koneksi ke endpoint gagal: {str(e)}")
            return None

def main():
    client = InferenceClient(endpoint_url=MODEL_URL, data_path=DATA_PATH)
    
    logging.info("Memulai proses inferensi manual...")
    payload = client.load_sample_data(sample_size=5)
    
    if payload:
        client.predict(payload)

if __name__ == "__main__":
    main()

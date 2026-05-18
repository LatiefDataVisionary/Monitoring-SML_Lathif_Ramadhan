import os
import time
import json
import logging
import random
import psutil
import requests
import pandas as pd
from prometheus_client import start_http_server, Counter, Gauge, Summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

MODEL_URL = "http://localhost:5000/invocations"
EXPORTER_PORT = 8001
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "MLProject_Folder", "data", "test_cleaned.csv")

# Definisi 10 Metrik Prometheus
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'Penggunaan CPU sistem dalam persen')
SYSTEM_RAM_USAGE = Gauge('system_ram_usage_percent', 'Penggunaan RAM sistem dalam persen')
NETWORK_RX_BYTES = Counter('network_receive_bytes_total', 'Total byte jaringan yang diterima')
NETWORK_TX_BYTES = Counter('network_transmit_bytes_total', 'Total byte jaringan yang dikirim')

MODEL_REQUESTS_TOTAL = Counter('model_requests_total', 'Total request yang dikirim ke model')
MODEL_ERRORS_TOTAL = Counter('model_errors_total', 'Total request yang menghasilkan error')
MODEL_LATENCY = Summary('model_request_latency_seconds', 'Waktu respons model dalam detik')

PREDICTION_CHURN_TOTAL = Counter('prediction_churn_total', 'Total prediksi pelanggan yang Churn (1)')
PREDICTION_NOT_CHURN_TOTAL = Counter('prediction_not_churn_total', 'Total prediksi pelanggan yang Tidak Churn (0)')
PREDICTION_PROBABILITY_AVG = Gauge('prediction_probability_average', 'Rata-rata probabilitas prediksi terakhir')

class MetricsCollector:
    """
    Kelas untuk mengumpulkan metrik sistem dan melakukan simulasi trafik ke model.
    """
    def __init__(self, endpoint_url: str, data_path: str):
        self.endpoint_url = endpoint_url
        self.data_path = data_path
        self.headers = {"Content-Type": "application/json"}
        self.dataset = self._initialize_dataset()
        
        self.net_io_initial = psutil.net_io_counters()

    def _initialize_dataset(self) -> pd.DataFrame:
        """
        Memuat dataset ke dalam memori untuk mempercepat proses sampling saat simulasi.
        """
        try:
            df = pd.read_csv(self.data_path)
            if 'Churn' in df.columns:
                df = df.drop(columns=['Churn'])
            logging.info("Dataset berhasil dimuat untuk simulasi trafik.")
            return df
        except Exception as e:
            logging.error(f"Gagal memuat dataset: {str(e)}")
            raise

    def update_system_metrics(self):
        """
        Memperbarui metrik infrastruktur (CPU, RAM, Network).
        """
        SYSTEM_CPU_USAGE.set(psutil.cpu_percent(interval=None))
        SYSTEM_RAM_USAGE.set(psutil.virtual_memory().percent)
        
        net_io_current = psutil.net_io_counters()
        rx_diff = net_io_current.bytes_recv - self.net_io_initial.bytes_recv
        tx_diff = net_io_current.bytes_sent - self.net_io_initial.bytes_sent
        
        NETWORK_RX_BYTES.inc(rx_diff)
        NETWORK_TX_BYTES.inc(tx_diff)
        
        self.net_io_initial = net_io_current

    def simulate_request(self):
        """
        Mengambil satu sampel data, mengirimkannya ke model, dan mencatat metrik performa.
        """
        sample_df = self.dataset.sample(n=1)
        payload = {
            "dataframe_split": {
                "columns": sample_df.columns.tolist(),
                "data": sample_df.values.tolist()
            }
        }

        MODEL_REQUESTS_TOTAL.inc()
        start_time = time.time()

        try:
            response = requests.post(self.endpoint_url, data=json.dumps(payload), headers=self.headers, timeout=5)
            latency = time.time() - start_time
            MODEL_LATENCY.observe(latency)

            if response.status_code == 200:
                prediction_result = response.json()
                
                # Asumsi output MLflow sklearn adalah list of integers, misal: [0] atau [1]
                if isinstance(prediction_result, list) and len(prediction_result) > 0:
                    predicted_class = prediction_result[0]
                    
                    if predicted_class == 1:
                        PREDICTION_CHURN_TOTAL.inc()
                        PREDICTION_PROBABILITY_AVG.set(1.0)
                    else:
                        PREDICTION_NOT_CHURN_TOTAL.inc()
                        PREDICTION_PROBABILITY_AVG.set(0.0)
                        
            else:
                MODEL_ERRORS_TOTAL.inc()
                logging.warning(f"Model mengembalikan status {response.status_code}")

        except requests.exceptions.RequestException:
            MODEL_ERRORS_TOTAL.inc()
            logging.error("Gagal terhubung ke endpoint model.")

def main():
    logging.info(f"Memulai Prometheus Exporter di port {EXPORTER_PORT}...")
    start_http_server(EXPORTER_PORT)
    
    collector = MetricsCollector(endpoint_url=MODEL_URL, data_path=DATA_PATH)
    
    logging.info("Memulai simulasi trafik. Tekan Ctrl+C untuk menghentikan.")
    try:
        while True:
            collector.update_system_metrics()
            collector.simulate_request()
            
            # Jeda acak antara 0.5 hingga 2 detik untuk mensimulasikan trafik dunia nyata
            sleep_duration = random.uniform(0.5, 2.0)
            time.sleep(sleep_duration)
            
    except KeyboardInterrupt:
        logging.info("Proses exporter dihentikan oleh pengguna.")

if __name__ == "__main__":
    main()

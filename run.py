import argparse
import glob
import os
import json

from data_streamer import DataStreamer
from data_analyzer import DataAnalyzer
from model_manager import ModelManager

from config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", choices=["update", "inference", "summary", "clear"])
    parser.add_argument("-file", type=str)
    args = parser.parse_args()

    streamer = DataStreamer()
    analyzer = DataAnalyzer()
    manager = ModelManager()

    if args.mode == "update":
        batch_file = streamer.get_next_batch()
        processed_file, _ = analyzer.analyze(batch_file)
        manager.train_and_evaluate(processed_file)

    elif args.mode == "inference":
        result = manager.predict(args.file)
        print(f"Путь к результатам: {result}")

    elif args.mode == "summary":
        metrics = sorted(glob.glob(f"{METADATA_DIR}/train_metrics_*"))
        if not metrics:
            print("Нет данных о запусках")
            return
        with open(metrics[-1], "r") as f:
            print(json.dumps(json.loads(f.read()), indent=4))


    elif args.mode == "clear":
        remove_list = (glob.glob(f"{METADATA_DIR}/*")
                       + glob.glob(f"{PROCESSED_DIR}/*")
                       + glob.glob(f"{RAW_DIR}/*")
                       + glob.glob(f"{MODELS_DIR}/*"))
        for f in remove_list:
            print(f"Remove {f}")
            os.remove(f)

if __name__ == "__main__":
    main()

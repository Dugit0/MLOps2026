import argparse
import glob
import os
import json
import shutil

from data_streamer import DataStreamer
from data_analyzer import DataAnalyzer
from model_manager import ModelManager

from config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", choices=["update",
                                          "inference",
                                          "summary",
                                          "clear",
                                          "get_batch"])
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
        latest_models = glob.glob(f"{MODELS_DIR}/latest/*")
        timestamps = []
        for name in latest_models:
            name, _, ext = name.partition('.')
            timestamp = '_'.join(name.split('_')[-2:])
            timestamps.append(timestamp)
        metrics = sorted(glob.glob(f"{METADATA_DIR}/train_metrics_*"))
        if not metrics:
            print("Нет данных о запусках")
            return
        print("Данные о последних обученных моделях:")
        for timestamp in timestamps:
            for metric in metrics:
                if timestamp in metric:
                    with open(metric) as f:
                        print(json.dumps(json.loads(f.read()), indent=4))

    elif args.mode == "clear":
        remove_list = (glob.glob(f"{METADATA_DIR}/*")
                       + glob.glob(f"{PROCESSED_DIR}/*")
                       + glob.glob(f"{RAW_DIR}/*")
                       + glob.glob(f"{MODELS_DIR}/*"))
        for f in remove_list:
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)
            print(f"Remove {f}")


    elif args.mode == "get_batch":
        batch_file = streamer.get_next_batch()
        processed_file, _ = analyzer.analyze(batch_file)
        print(processed_file)

    else:
        print(f"Неизвестное значение параметра -mode={args.mode}")

if __name__ == "__main__":
    main()

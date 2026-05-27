from pathlib import Path

from src.train import train_model


BASE_DIR = Path(__file__).resolve().parent


if __name__ == '__main__':
    meta = train_model(
        csv_path=BASE_DIR / 'data/sample_train.csv',
        model_dir=BASE_DIR / 'models',
        chunksize=50,
    )
    print(meta)

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lse_vision.config.settings import AppConfig, load_config
from lse_vision.dataset.sign_image_dataset import discover_samples
from lse_vision.utils.logging_utils import configure_logging, get_logger
from lse_vision.utils.runtime import collect_runtime_info, ensure_project_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LSE Sign Classifier — entrypoint unificado",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=[
            "info",             # muestra info del dataset
            "train",            # entrena el clasificador CNN
            "evaluate",         # evalúa el CNN en el test set
            "predict",          # predice una imagen
            "web",              # lanza la aplicación web
            "capture",          # captura landmarks con la webcam (guiado letra a letra)
            "train-landmarks",  # entrena el MLP de landmarks
            "extract-video",    # etiqueta landmarks desde un fichero de vídeo
            "extract-dataset",  # extrae landmarks del dataset de imágenes existente
            "capture-images",   # captura fotos reales con la webcam letra a letra
        ],
        default="info",
        help=(
            "info             — muestra clases e imágenes del dataset\n"
            "train            — entrena la CNN (MobileNetV2 por defecto)\n"
            "evaluate         — evalúa la CNN en el test set\n"
            "predict          — predice letra de una imagen  (requiere --image)\n"
            "web              — lanza la aplicación web\n"
            "capture          — captura tus propios gestos letra a letra\n"
            "train-landmarks  — entrena el MLP con los landmarks capturados\n"
            "extract-video    — etiqueta landmarks desde un vídeo  (requiere --video)\n"
            "extract-dataset  — extrae landmarks de las imágenes del dataset automáticamente\n"
        ),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None, help="Ruta de imagen para --mode predict")
    parser.add_argument("--epochs", type=int, default=None, help="Override de épocas para --mode train")
    parser.add_argument("--video", type=Path, default=None, help="Ruta de vídeo para --mode extract-video")
    parser.add_argument("--csv", type=Path, default=None, help="CSV de landmarks (capture / train-landmarks)")
    parser.add_argument("--letters", type=str, default=None,
                        help="Letras a capturar, p.ej. 'ABCDE' (sólo para --mode capture)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host para --mode web")
    parser.add_argument("--port", type=int, default=8000, help="Puerto para --mode web")
    return parser.parse_args()


# ─── Mode runners ─────────────────────────────────────────────────────────────

def run_info_mode(config: AppConfig) -> int:
    logger = get_logger(__name__)
    ensure_project_directories(config)
    samples, classes = discover_samples(
        dataset_root=config.paths.dataset_root,
        allowed_extensions=config.data.allowed_extensions,
    )
    runtime_info = collect_runtime_info(config)
    logger.info("Detectadas %s clases y %s imágenes.", len(classes), len(samples))
    payload = {
        "project_name": config.project_name,
        "dataset_root": str(config.paths.dataset_root),
        "num_classes": len(classes),
        "classes": classes,
        "num_images": len(samples),
        "image_size": config.data.image_size,
        "architecture": config.model.architecture,
        "device": runtime_info["device"],
        "directories": runtime_info["directories"],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def run_train_mode(config: AppConfig, args: argparse.Namespace) -> int:
    from lse_vision.training.train import run_training
    return run_training(config, epochs_override=args.epochs)


def run_evaluate_mode(config: AppConfig, args: argparse.Namespace) -> int:
    from lse_vision.training.evaluate import run_evaluate
    ckpt = args.checkpoint or (config.paths.checkpoints_dir / "best_static_sign_cnn.pt")
    return run_evaluate(config, checkpoint_path=ckpt)


def run_predict_mode(config: AppConfig, args: argparse.Namespace) -> int:
    if args.image is None:
        print("ERROR: --image es obligatorio para el modo predict", file=sys.stderr)
        return 1
    from lse_vision.inference.predict import run_predict
    ckpt = args.checkpoint or (config.paths.checkpoints_dir / "best_static_sign_cnn.pt")
    return run_predict(config, image_path=args.image, checkpoint_path=ckpt)


def run_web_mode(config: AppConfig, args: argparse.Namespace) -> int:
    import uvicorn
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    print(f"\n  LSE Vision -> http://{host}:{port}\n")
    uvicorn.run("lse_vision.web.app:app", host=host, port=port, reload=False)
    return 0


def run_capture_mode(config: AppConfig, args: argparse.Namespace) -> int:
    from lse_vision.capture.landmark_capture import run_capture
    csv_path = args.csv or (config.paths.data_dir / "landmarks.csv")
    letters = list(args.letters.upper()) if args.letters else None
    return run_capture(config, output_csv=csv_path, letters=letters)


def run_train_landmarks_mode(config: AppConfig, args: argparse.Namespace) -> int:
    from lse_vision.training.train_landmarks import run_train_landmarks
    return run_train_landmarks(config, csv_path=args.csv)


def run_extract_video_mode(config: AppConfig, args: argparse.Namespace) -> int:
    if args.video is None:
        print("ERROR: --video es obligatorio para el modo extract-video", file=sys.stderr)
        print("  Descarga el vídeo con:  pip install yt-dlp  &&  yt-dlp <URL> -o video.mp4")
        return 1
    from lse_vision.capture.video_extractor import run_extract_video
    csv_path = args.csv or (config.paths.data_dir / "landmarks.csv")
    return run_extract_video(config, video_path=args.video, output_csv=csv_path)


def run_extract_dataset_mode(config: AppConfig, args: argparse.Namespace) -> int:
    from lse_vision.capture.dataset_extractor import run_extract_dataset
    csv_path = args.csv or (config.paths.data_dir / "landmarks.csv")
    return run_extract_dataset(config, output_csv=csv_path)


# ─── Dispatcher ───────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config.paths.logs_dir)

    dispatch = {
        "info": run_info_mode,
        "train": run_train_mode,
        "evaluate": run_evaluate_mode,
        "predict": run_predict_mode,
        "web": run_web_mode,
        "capture": run_capture_mode,
        "train-landmarks": run_train_landmarks_mode,
        "extract-video": run_extract_video_mode,
        "extract-dataset": run_extract_dataset_mode,
    }

    fn = dispatch[args.mode]
    # info mode only takes config; others take config + args
    if args.mode == "info":
        return fn(config)
    return fn(config, args)


if __name__ == "__main__":
    sys.exit(main())

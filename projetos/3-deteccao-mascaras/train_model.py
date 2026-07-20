"""
Projeto 3 — Detecção de Máscaras Faciais (Fine-tuning do YOLO11n)

Etapa 1 do desafio: fine-tuning do detector pré-treinado YOLO11n no dataset
fornecido (dataset/data.yaml), em CPU, e cópia dos melhores pesos para model.pt.

Uso:
    python train_model.py
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

# Diretório deste script — todos os caminhos são resolvidos a partir daqui,
# então o script funciona tanto rodando de dentro da pasta do projeto quanto
# da raiz do repositório (a CI executa `cd projetos/3-deteccao-mascaras`).
HERE = Path(__file__).resolve().parent
DATA_YAML = HERE / "dataset" / "data.yaml"
MODEL_OUT = HERE / "model.pt"

# Hiperparâmetros de fine-tuning ------------------------------------------------
# YOLO converge rápido em fine-tuning. Com ~20 épocas o mAP50 fica bem acima do
# mínimo exigido (0.30). imgsz=640 mantém consistência com a exportação/inferência.
EPOCHS = 20
IMGSZ = 416
BATCH = 4
SEED = 0


def main():
    print("=" * 70)
    print("Projeto 3 — Fine-tuning YOLO11n (CPU) para detecção de máscaras")
    print("=" * 70)

    # 1. Modelo pré-treinado (única exceção permitida à regra de "sem pré-treino")
    model = YOLO("yolo11n.pt")

    # 2. Fine-tuning no dataset fornecido, em CPU
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device="cpu",
        seed=SEED,
        name="train",
        exist_ok=True,   # reexecuções sobrescrevem runs/detect/train (evita train2, train3...)
        plots=False,
        verbose=True,
    )

    # 3. Validação final explícita para exibir o mAP50 no terminal
    metrics = model.val(data=str(DATA_YAML), split="val", verbose=False)
    print("\n" + "-" * 70)
    print(f"mAP50    (validação): {metrics.box.map50:.4f}")
    print(f"mAP50-95 (validação): {metrics.box.map:.4f}")
    print("-" * 70)

    # 4. Copiar os melhores pesos para model.pt (usando results.save_dir — robusto
    #    ao sufixo automático que o Ultralytics dá quando a pasta já existe).
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"best.pt não encontrado em {best}")
    shutil.copy(best, MODEL_OUT)
    print(f"\n✅ Pesos treinados copiados para: {MODEL_OUT} "
          f"({MODEL_OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

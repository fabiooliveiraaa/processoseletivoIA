"""
Projeto 3 — Otimização do Modelo (Exportação para Edge / TFLite)

Etapa 2 do desafio: carrega model.pt e exporta para TensorFlow Lite (LiteRT),
gerando model.tflite na raiz desta pasta — o artefato de edge que a correção
automática valida.

Uso:
    python optimize_model.py
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
MODEL_PT = HERE / "model.pt"
MODEL_TFLITE = HERE / "model.tflite"
IMGSZ = 640  # mesmo tamanho usado no treino e na inferência


def main():
    print("=" * 70)
    print("Projeto 3 — Exportação para TFLite (Edge AI)")
    print("=" * 70)

    if not MODEL_PT.is_file():
        raise FileNotFoundError(
            f"{MODEL_PT} não encontrado. Rode train_model.py antes de otimizar."
        )

    # 1. Carregar os pesos treinados
    model = YOLO(str(MODEL_PT))

    # 2. Exportar para TFLite. Em Ultralytics >= 8.4, format="tflite" é
    #    redirecionado para o export unificado LiteRT e gera "model.tflite"
    #    ao lado de model.pt (mesmo stem do arquivo de origem).
    exported = model.export(format="tflite", imgsz=IMGSZ)

    # 3. Garantir, de forma robusta, que exista "model.tflite" na raiz da pasta.
    #    (Se por qualquer motivo o arquivo sair com outro nome/local, copiamos.)
    exported_path = Path(exported) if exported else None
    if exported_path and exported_path.is_file() and exported_path.resolve() != MODEL_TFLITE.resolve():
        shutil.copy(exported_path, MODEL_TFLITE)

    if not MODEL_TFLITE.is_file():
        # Último recurso: procurar qualquer .tflite gerado (exceto edgetpu)
        candidates = [
            p for p in HERE.glob("*.tflite")
            if "edgetpu" not in p.name
        ]
        if candidates:
            shutil.copy(candidates[0], MODEL_TFLITE)

    if not MODEL_TFLITE.is_file():
        raise FileNotFoundError(
            "Falha ao gerar model.tflite. Verifique a saída da exportação acima."
        )

    # 4. Limpar artefatos intermediários para manter a pasta limpa no commit
    for junk in ["model.onnx", "model_saved_model"]:
        p = HERE / junk
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)

    pt_kb = MODEL_PT.stat().st_size / 1024
    tflite_kb = MODEL_TFLITE.stat().st_size / 1024
    print("\n" + "-" * 70)
    print(f"model.pt     : {pt_kb:8.1f} KB")
    print(f"model.tflite : {tflite_kb:8.1f} KB")
    print("-" * 70)
    print(f"\n✅ Modelo de edge gerado em: {MODEL_TFLITE}")


if __name__ == "__main__":
    main()

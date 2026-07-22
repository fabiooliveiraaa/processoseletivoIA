## 📝 Relatório do Candidato

👤 **Nome Completo:** [SEU NOME COMPLETO AQUI]

### 1️⃣ Resumo da Abordagem

Utilizei **fine-tuning** do detector pré-treinado **YOLO11n** (variante *nano*,
indicada para CPU/edge) sobre o dataset fornecido de detecção de máscaras (3
classes: `with_mask`, `without_mask`, `mask_weared_incorrect`). Parti dos pesos
pré-treinados no COCO (`yolo11n.pt`) e os adaptei às 3 classes do problema,
aproveitando as *features* de baixo/médio nível já aprendidas e reduzindo o
número de épocas necessárias em relação a um treino do zero.

Hiperparâmetros de fine-tuning e a justificativa técnica de cada escolha:

- **Épocas = 20:** YOLO converge rápido em fine-tuning. O mAP50 já ultrapassou 0.30 (o mínimo exigido) logo na 1ª época e estabilizou perto de 0.72 ao fim das 20, sem sinais de overfitting.
- **imgsz = 416:** reduzido de 640 para 416 para caber na memória RAM do ambiente de execução (GitHub Codespaces/Actions) e acelerar o treino em CPU, mantendo resolução suficiente para os rostos.
- **batch = 4:** foi decisivo. Com batch 16 o processo era encerrado por falta de memória (`Terminated`); com 4 o uso de RAM estabilizou sem prejudicar a convergência.
- **device = cpu:** restrição do desafio (treino apenas em CPU).
- **optimizer = auto (→ AdamW):** deixei o Ultralytics escolher automaticamente, que selecionou AdamW com `lr0 ≈ 0.00143`, adequado para um fine-tuning estável.
- **seed = 0:** para reprodutibilidade entre execuções.

O dataset é **desbalanceado**: `mask_weared_incorrect` tem pouquíssimos exemplos
(apenas 19 instâncias na validação, contra 593 de `with_mask`). Mantive a
estratégia padrão do YOLO sem reponderação manual, já que o objetivo é demonstrar
o pipeline completo, e registro o impacto desse desbalanceamento na seção 4.

### 2️⃣ Bibliotecas Utilizadas

- **ultralytics 8.4.102** — framework YOLO (treino, validação, exportação)
- **torch 2.12.1** — backend de deep learning
- **torchvision 0.27.1** — operações de visão computacional (dependência do YOLO)
- **litert-torch 0.9.1** — conversão PyTorch → TFLite/LiteRT
- **opencv-python-headless** — processamento de imagem sem dependência de libGL

As versões de `torch`/`torchvision`/`litert-torch` foram **fixadas** no
`requirements.txt` de propósito (ver seção 5).

### 3️⃣ Técnica de Otimização do Modelo

A otimização para edge foi feita **exportando o modelo treinado (`model.pt`)
para o formato TensorFlow Lite / LiteRT** via `model.export(format="tflite")`.
A partir da versão 8.4.83 do Ultralytics, `format="tflite"` é automaticamente
redirecionado para o export unificado **LiteRT** (aviso emitido no log), que gera
o arquivo `model.tflite` — um *flatbuffer* único, autocontido e pronto para
execução em dispositivos de borda (celulares, SBCs, sistemas embarcados com
runtime LiteRT/XNNPACK), sem depender do stack completo do PyTorch em produção.

O modelo foi exportado em precisão **float32**, priorizando a robustez e a
fidelidade numérica da conversão (o mAP50 do `.tflite` fica praticamente idêntico
ao do `.pt`). Como passo adicional de compressão, o fluxo LiteRT também oferece
**quantização dinâmica INT8** (`quantize="w8a32"`: pesos em int8, ativações em
FP32), que reduziria o tamanho dos pesos ao custo de uma pequena variação de
acurácia.

### 4️⃣ Resultados Obtidos

Métricas de detecção no conjunto de validação (model.pt):
- **Geral (todas as classes):** mAP50 = 0.726 | mAP50-95 = 0.504
- **with_mask:** mAP50 = 0.947 | mAP50-95 = 0.647
- **without_mask:** mAP50 = 0.718 | mAP50-95 = 0.453
- **mask_weared_incorrect:** mAP50 = 0.504 | mAP50-95 = 0.405

Tamanho dos artefatos:
- **model.pt:** 5303.8 KB (~5.2 MB)
- **model.tflite:** 10379.4 KB (~10.1 MB)

Discussão: o `model.pt` tem desempenho muito bom na classe majoritária
`with_mask` (mAP50 0.947) e razoável em `without_mask` (0.718). Como esperado, a
classe minoritária `mask_weared_incorrect` teve o pior desempenho (mAP50 0.504) —
reflexo direto do forte desbalanceamento do dataset (apenas 19 instâncias na
validação). Sobre os tamanhos, o `.tflite` ficou **maior** que o `.pt` porque a
exportação LiteRT padrão grava os pesos em float32, enquanto o `.pt` os armazena
de forma mais compacta; a otimização aqui é a conversão para um formato de runtime
de borda, não a compressão por quantização.

### 5️⃣ Comentários Adicionais

- **Dificuldade real — conflito de versões na exportação:** a maior dificuldade
  não foi o treino, e sim a exportação para TFLite. A ferramenta LiteRT
  (`litert-torch`) exige `torch >=2.4,<2.13` em conjunto com `torchao>=0.17`.
  Com o `torch 2.13` (padrão do ambiente) a exportação falhava com
  `ImportError: cannot import name 'get_cuda_generator_meta_val'`, e com o `2.4.1`
  falhava com `ValueError: infer_schema (...) unsupported default value`. A
  solução foi **fixar `torch==2.12.1`** e **declarar `litert-torch` no
  `requirements.txt`**, garantindo que as versões corretas sejam instaladas
  *antes* de o script rodar (instalar via AutoUpdate no meio da execução não tem
  efeito, pois o torch já estava carregado em memória).
- **Decisão técnica — memória:** reduzi `batch` (16 → 4) e `imgsz` (640 → 416)
  após o treino ser encerrado por estouro de RAM (`Terminated`) no ambiente CPU.
- **Robustez do pipeline:** uso `results.save_dir` para localizar o `best.pt`
  (em vez de um caminho fixo `runs/detect/train/`, que quebraria em reexecuções),
  e o `optimize_model.py` garante que `model.tflite` exista na raiz e remove
  artefatos intermediários, mantendo o commit limpo.
- **Limitação do modelo:** por ser `nano` e treinado em baixa resolução (416) e
  poucas épocas, o modelo tende a errar mais em rostos pequenos/distantes e na
  classe minoritária. Para produção, aumentaria `imgsz`, épocas e aplicaria
  técnicas de balanceamento (oversampling ou *class weights*).

### 6️⃣ Exemplo de Inferência

Saída do terminal ao rodar `python run_inference.py` sobre 5 imagens de validação,
uma de cada vez (cenário real de edge, batch=1):

```
Imagem                               Detecções  Detalhes
----------------------------------------------------------------------
maksssksksss105.jpg                          9  [9x with_mask]
maksssksksss107.jpg                          1  [1x with_mask]
maksssksksss11.jpg                          22  [1x mask_weared_incorrect, 21x with_mask]
maksssksksss113.jpg                          3  [3x with_mask]
maksssksksss12.jpg                          14  [11x with_mask, 3x without_mask]
----------------------------------------------------------------------
TOTAL                                       49
```

Caso observado: na imagem `maksssksksss11.jpg`, uma cena com muitas pessoas, o
modelo detectou 22 rostos e — apesar do forte desbalanceamento — conseguiu
identificar **1 ocorrência da classe minoritária `mask_weared_incorrect`**, o que
mostra que ele aprendeu a classe, ainda que com menor recall. Já em
`maksssksksss12.jpg` distinguiu corretamente pessoas com e sem máscara (11x
`with_mask` e 3x `without_mask`). Ao abrir as imagens anotadas em
`runs/detect/inferencia_exemplos/predicoes/`, as *bounding boxes* estavam bem
posicionadas sobre os rostos, com maior confiança na classe majoritária.
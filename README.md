##  Relatório do Candidato

 **Nome Completo:** [SEU NOME COMPLETO AQUI]

###  Resumo da Abordagem

Utilizei **fine-tuning** do detector pré-treinado **YOLO11n** (variante *nano*,
indicada para CPU/edge) sobre o dataset fornecido de detecção de máscaras (3
classes: `with_mask`, `without_mask`, `mask_weared_incorrect`). Os pesos
pré-treinados no COCO foram adaptados às 3 classes do problema, aproveitando as
features de baixo/médio nível já aprendidas e reduzindo o número de épocas
necessárias.

Hiperparâmetros de fine-tuning:
- **Épocas:** 20
- **Tamanho de imagem (imgsz):** 416
- **Batch size:** 4
- **Dispositivo:** CPU (`device="cpu"`)
- **Otimizador:** AdamW (selecionado automaticamente pelo `optimizer="auto"` do
  Ultralytics), com data augmentation padrão do pipeline YOLO (mosaic, flip,
  HSV, etc.).

O `batch` e o `imgsz` foram reduzidos (de 16/640 para 4/416) para caber na
memória RAM disponível no ambiente de execução (GitHub Codespaces), sem prejuízo
para o objetivo do desafio — o mAP50 obtido ficou bem acima do mínimo exigido.

Sobre o **desbalanceamento de classes**: a classe `mask_weared_incorrect` tem
muito menos exemplos que as outras duas, o que se reflete em um desempenho
inferior nela. Mantive a estratégia padrão do YOLO sem reponderação manual, já
que o objetivo do desafio é demonstrar o pipeline completo (fine-tuning →
validação → exportação) funcionando corretamente.

###  Bibliotecas Utilizadas

- **ultralytics** 8.4.102 (framework YOLO: treino, validação, exportação)
- **torch** 2.12.1 / **torchvision** 0.27.1 (backend de deep learning)
- **opencv-python-headless** (processamento de imagem, sem dependência de libGL)
- Para a exportação TFLite, o Ultralytics instala automaticamente o fluxo
  unificado **LiteRT** (litert-torch, ai-edge-litert, torchao).

###  Técnica de Otimização do Modelo

A otimização para edge foi feita **exportando o modelo treinado (`model.pt`)
para o formato TensorFlow Lite / LiteRT** via `model.export(format="tflite")`.
A partir da versão 8.4.83 do Ultralytics, `format="tflite"` é redirecionado para
o export unificado **LiteRT**, que gera o arquivo `model.tflite` — um flatbuffer
único, autocontido e pronto para execução em dispositivos de borda (celulares,
SBCs, sistemas embarcados com runtime LiteRT), sem depender do stack completo do
PyTorch em produção. O modelo foi exportado em float32; como o critério do
projeto não exige redução de tamanho, priorizei a robustez da conversão. Um
passo adicional possível seria a quantização dinâmica INT8, que reduziria o
tamanho dos pesos ao custo de uma pequena variação de acurácia.

###  Resultados Obtidos

| Métrica / Artefato | Valor |
| --- | --- |
| mAP50 (validação, `model.pt`) | 0.7157 |
| mAP50-95 (validação, `model.pt`) | 0.5021 |
| Tamanho `model.pt` | 5303.7 KB (~5.3 MB) |
| Tamanho `model.tflite` | 10379.4 KB (~10.1 MB) |

Observação: o `model.tflite` ficou maior que o `model.pt` porque a exportação
padrão LiteRT grava os pesos em float32, enquanto o `.pt` os armazena de forma
mais compacta. A otimização aqui é a conversão para um formato de runtime de
borda, não a compressão por quantização.

###  Comentários Adicionais (Opcional)

- **Reprodutibilidade:** fixei `seed=0` e uso `results.save_dir` para localizar o
  `best.pt`, evitando depender de um caminho fixo `runs/detect/train/...` que
  quebraria em reexecuções (o Ultralytics cria `train2`, `train3`, etc.).
- **Robustez da exportação:** o `optimize_model.py` garante que `model.tflite`
  exista na raiz da pasta e remove artefatos intermediários para manter o commit
  limpo. Foi necessário fixar `torch` na faixa `>=2.9,<2.13` no `requirements.txt`,
  pois a ferramenta de exportação LiteRT (litert-torch) exige essa janela de
  versão em conjunto com `torchao>=0.17`.
- **Desbalanceamento:** conforme o README, a classe `mask_weared_incorrect` é a
  minoritária. Ainda assim, o modelo conseguiu detectar 1 ocorrência dessa classe
  na imagem de teste `maksssksksss11.jpg`, o que mostra que ele aprendeu a classe,
  ainda que com menos confiança que as demais.

### Exemplo de Inferência

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

Ao abrir as imagens anotadas em `runs/detect/inferencia_exemplos/predicoes/`,
observei que as caixas ficaram bem localizadas sobre os rostos. O modelo se saiu
muito bem na classe majoritária `with_mask` e identificou corretamente pessoas
`without_mask` na imagem `maksssksksss12.jpg`. A classe minoritária
`mask_weared_incorrect` apareceu apenas uma vez (imagem `maksssksksss11.jpg`),
condizente com o forte desbalanceamento do dataset.
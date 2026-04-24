Coloque os datasets exportados e os modelos TFLite nesta pasta `assets`.

Estrutura esperada:

- `export_cifar10/labels.csv`
- `export_cifar10/images_png/*.png`
- `export_cifar100/labels.csv`
- `export_cifar100/images_png/*.png`
- `export_wakevision/labels.csv`
- `export_wakevision/images_png/*.png`
- `*.tflite` (na raiz de `assets` ou ajuste no app)

Regras da avaliação no app:

- A execução é por dataset selecionado (um por rodada).
- O app usa mapeamento pareado automático por nome do arquivo do modelo:
  - dataset `cifar10` -> modelos contendo `cifar10`
  - dataset `cifar100` -> modelos contendo `cifar100`
  - dataset `wakevision` -> modelos contendo `wakevision`, `wake_vision` ou `wake`
- Há switch para usar ou não aceleração de hardware:
  - desligado: `CPU`
  - ligado: tenta `NNAPI`, com fallback para `CPU_FALLBACK` se necessário

Campos relevantes no JSON de saída por modelo:

- `accuracy`
- `avgLatencyMs`, `p90LatencyMs`, `throughputImagesPerSecond`
- `useHardwareAcceleration`
- `effectiveDelegate`
- `evalStatus`
- `errorMessage`

Exemplo de origem dos exports:

- `C:\Users\saulo\PIBIC\Lightweight Analyses\export_cifar10`
- `C:\Users\saulo\PIBIC\Lightweight Analyses\export_cifar100`
- `C:\Users\saulo\PIBIC\Lightweight Analyses\export_wakevision`


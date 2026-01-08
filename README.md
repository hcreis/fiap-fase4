# Análise de Vídeo — Tech Challenger Fase 4 🔍

**Resumo:** Projeto que processa um vídeo para detectar faces, estimar emoções, estimar pose (atividades) e gerar relatórios por cena. O script principal é `tech_challenger_fase_4.py`.

---

## ✅ Requisitos

- **Python 3.11** (OBRIGATÓRIO)
- Sistema operacional: Linux (testado) — outras plataformas podem funcionar, ajustes de dependências podem ser necessários
- **TensorFlow** (necessário para `deepface` — instale `tensorflow` ou `tensorflow-cpu` conforme disponível)
- Espaço para modelos e vídeo de entrada

## 📦 Instalação

1. Crie e ative um ambiente virtual com Python 3.11 (recomendado):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

2. Instale dependências:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Dica: se ocorrerem erros com bibliotecas C (ex.: OpenCV), verifique que você tem ferramentas de build e dependências do sistema instaladas.

## 🗂️ Arquivos importantes

- `tech_challenger_fase_4.py` — script principal
- `requirements.txt` — dependências do projeto
- `yolo11n-pose.pt` — modelo YOLO de pose (deve estar no diretório do projeto)
- `video_tech_challenger_fase_4.mp4` — vídeo de entrada (coloque no diretório do projeto)
- `blaze_face_short_range.tflite` — arquivo TFLite incluído no repositório

## 📥 Modelos necessários

- - Emoções: o script usa `DeepFace.analyze` (backend TensorFlow) para estimar emoções.
- `yolo11n-pose.pt` — modelo de pose usado pelo script; o arquivo `yolo11n-pose.pt` está incluído no repositório atual.
- O InsightFace (`buffalo_l`) é carregado automaticamente pela biblioteca InsightFace (será feito download quando necessário, se houver internet).

## ▶️ Como executar

Coloque o vídeo de entrada com o nome `video_tech_challenger_fase_4.mp4` na raiz do projeto ou edite o `main()` para apontar para outro arquivo.

Execute:

```bash
python tech_challenger_fase_4.py
```

O script irá gerar no mesmo diretório:

- `video_tech_challenger_fase_4_final.mp4` — vídeo anotado
- `relatorio_final_tecnico.txt` — relatório técnico
- `grafico_emocoes.png` — gráfico com a distribuição de emoções (se houver detecções)

## ⚙️ Configurações úteis

No topo de `tech_challenger_fase_4.py` há várias constantes que você pode adaptar:

- `PULAR_FRAMES` — pular frames para acelerar processamento (0 = desativado)
- `LIMIAR_SIMILARIDADE_FACE` — quão rígida é a fusão de embeddings faciais
- `MIN_FRAMES_TROCA_CENA` — janela mínima para trocar de cena
- `MIN_FRAMES_PARA_CONFIRMAR` — quantos frames para confirmar um aceno

## 🐞 Solução de problemas

- Erro: `Não foi possível abrir o vídeo` → confirme o caminho e o nome do arquivo (`video_tech_challenger_fase_4.mp4`) e codecs.
- Erro relacionado a `yolo11n-pose.pt` → coloque o arquivo correto na raiz ou altere para um checkpoint disponível.
- Lentidão / uso alto de CPU → o pipeline roda em CPU (intencional). Para acelerar, use máquinas com CPU mais rápida ou GPU e adapte os providers do ONNX/InsightFace (requer drivers CUDA e builds compatíveis).
- Projeto testado e valido com CPU.

## 👥 Membros

- Helen de Cassia dos Reis Cruz | RM364533
- Leandro Bernardo dos Santos | RM364501

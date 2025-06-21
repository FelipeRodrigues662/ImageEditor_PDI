# Sistema de Processamento Digital de Imagens

## Descrição
Sistema interativo para edição e análise de imagens desenvolvido para a disciplina SIN 392 - Introdução ao Processamento Digital de Imagens da Universidade Federal de Viçosa Campus Rio Paranaíba.

## Funcionalidades Implementadas

### 1. Histograma
- Cálculo e exibição do histograma da imagem
- Visualização em tempo real

### 2. Transformações de Intensidade
- Alargamento de Contraste
- Equalização de Histograma

### 3. Filtros Passa-Baixa
- Média
- Mediana
- Gaussiano
- Máximo
- Mínimo

### 4. Filtros Passa-Alta
- Laplaciano
- Roberts
- Prewitt
- Sobel

### 5. Convolução no Domínio da Frequência
- Filtros passa-alta e passa-baixa
- Transformada de Fourier

### 6. Espectro de Fourier
- Exibição da imagem com seu espectro de Fourier

### 7. Morfologia Matemática
- Operações de erosão e dilatação

### 8. Segmentação
- Método de Otsu para limiarização

### 9. Funcionalidades Extras
- Sistema de desfazer/refazer (Ctrl+Z/Ctrl+Y)
- Carregamento e salvamento de imagens
- Suporte a imagens RGB e níveis de cinza
- Interface gráfica intuitiva

## Requisitos do Sistema

### Software Necessário
- Python 3.8 ou superior
- Anaconda (recomendado)

### Bibliotecas Python
- tkinter (incluída com Python)
- numpy
- opencv-python
- matplotlib
- scipy
- scikit-image
- pillow

## Instalação e Configuração

### 1. Clone o Repositório
```bash
git clone https://github.com/FelipeRodrigues662/ImageEditor_PDI
cd PDI
```

### 2. Criação do Ambiente Conda
```bash
conda create -n pdi_env python=3.9
conda activate pdi_env
```

### 3. Instalação das Dependências
```bash
pip install -r requirements.txt
```

### 4. Execução do Sistema
```bash
python main.py
```

## Como Usar

### Carregamento de Imagem
1. Clique em "Abrir Imagem" ou use Ctrl+O
2. Selecione uma imagem (formato: JPG, PNG, BMP, TIFF)
3. A imagem será automaticamente convertida para níveis de cinza se necessário

### Aplicação de Filtros
1. Selecione o filtro desejado no menu lateral
2. Ajuste os parâmetros conforme necessário
3. Clique em "Aplicar" para processar a imagem

### Sistema de Desfazer/Refazer
- **Ctrl+Z**: Desfazer última ação
- **Ctrl+Y**: Refazer ação desfeita

### Salvamento
- Use "Salvar Imagem" ou Ctrl+S para salvar a imagem processada

## Estrutura do Projeto

```
PDI/
├── main.py                 # Arquivo principal
├── requirements.txt        # Dependências
├── README.md              # Este arquivo
├── src/
│   ├── gui/
│   │   ├── main_window.py
│   │   ├── image_viewer.py
│   │   └── controls.py
│   ├── processing/
│   │   ├── histogram.py
│   │   ├── intensity.py
│   │   ├── filters.py
│   │   ├── fourier.py
│   │   ├── morphology.py
│   │   └── segmentation.py
│   └── utils/
│       ├── image_utils.py
│       └── history.py
```

## Desenvolvedor
- Disciplina: SIN 392 - Introdução ao Processamento Digital de Imagens
- Universidade Federal de Viçosa Campus Rio Paranaíba
- Período: 2025-1

## Licença
Este projeto foi desenvolvido para fins educacionais. 

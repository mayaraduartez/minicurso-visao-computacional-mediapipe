# 👁️ Mini Curso — Visão Computacional com MediaPipe

Mini curso introdutório de Visão Computacional utilizando **Python**, **OpenCV** e **MediaPipe** para reconhecimento Facial e de Mãos.

O projeto apresenta exemplos práticos de:
- reconhecimento facial;
- detecção de mãos;
- contador de dedos;
- filtros faciais em tempo real.

---

# 🚀 Tecnologias Utilizadas

- Python 3
- OpenCV
- MediaPipe
- NumPy

---

# 📦 Instalação

Clone o repositório:

```bash
git clone https://github.com/SEU-USUARIO/minicurso-mediapipe.git
```

Entre na pasta:

```bash
cd minicurso-mediapipe
```

Crie um ambiente virtual:

```bash
python -m venv venv
```

Ative o ambiente virtual:

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Instale as dependências:

```bash
pip install opencv-python mediapipe numpy
```

---

# 📁 Estrutura do Projeto

```text
models/
 ├── face_landmarker.task
 └── hand_landmarker.task

face.py
maos.py
contador.py
volume.py
filtro.py
```

---

# 🧠 Conteúdos do Mini Curso

## 📍 Introdução à Visão Computacional

- O que é visão computacional
- Aplicações no dia a dia
- Introdução ao MediaPipe

---

## 😀 Reconhecimento Facial

- Face Mesh
- Landmarks faciais
- Coordenadas X, Y e Z

---

## ✋ Detecção de Mãos

- Hand Landmarker
- Rastreamento das mãos
- Desenho dos landmarks

---

## ✌️ Contador de Dedos

- Identificação dos dedos levantados
- Comparação de landmarks
- Contagem em tempo real

---

## 🥸 Filtro Facial

- Óculos desenhados em tempo real
- Nariz estilizado
- Uso de coordenadas faciais

---

# 📌 Modelos `.task`

Os modelos utilizados podem ser baixados no site oficial do Google Dev:

- Face Landmarker
- Hand Landmarker

🔗 https://ai.google.dev/edge/mediapipe/solutions/guide

---

# 🎯 Objetivo

Demonstrar de forma simples e prática como utilizar visão computacional em projetos interativos utilizando Python e MediaPipe.

---

# 👨‍💻 Autor

Mayara Duarte  
Mini Curso — Visão Computacional com MediaPipe

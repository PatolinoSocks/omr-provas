# OMR-Provas

Sistema de correção automática de provas objetivas utilizando visão computacional.

O sistema detecta automaticamente as bolhas preenchidas em gabaritos e gera a nota do aluno.

## Funcionalidades

- Correção automática por foto
- Processamento em lote de várias provas
- Exportação para Excel ou CSV
- Aplicação web para professores (Streamlit)
- Suporte até 40 questões

## Estrutura do projeto

app.py
Aplicação web para correção das provas

omr_engine.py
Motor de visão computacional responsável pela detecção das bolhas

run_batch.py
Script para corrigir várias provas em lote

dev_notebook.ipynb
Notebook usado para desenvolvimento e testes

requirements.txt
Dependências do projeto


## Como rodar localmente

Instale as dependências:

pip install -r requirements.txt


Rodar aplicação web:

streamlit run app.py


## Uso

1. Tire foto do gabarito
2. Envie a imagem no aplicativo
3. Cole o gabarito da prova
4. O sistema gera automaticamente a nota e um arquivo Excel com os resultados.

## Autor

Raphael Alvim
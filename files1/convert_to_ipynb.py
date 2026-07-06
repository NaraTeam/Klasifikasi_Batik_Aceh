import json
import re

with open('train.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Split by the header comments
parts = re.split(r'(?=# ======================================================)', content)

cells = []
for part in parts:
    if part.strip():
        lines = [line + '\n' for line in part.split('\n')]
        if lines:
            lines[-1] = lines[-1][:-1]
        
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

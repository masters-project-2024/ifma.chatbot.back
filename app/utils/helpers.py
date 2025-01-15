from fastapi import HTTPException
from typing import Dict

# Definindo uma função para validar a mensagem
def validate_message(data: Dict) -> str:
    if 'text' not in data:
        raise HTTPException(status_code=400, detail="Invalid message format")
    return data['text']
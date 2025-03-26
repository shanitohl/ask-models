# API RAG - Asesor OSCE

## Create a virtual environment
python3 -m venv _env

## Activate the virtual environment
source _env/bin/activate


## Requisitos
Instala los paquetes necesarios:
```
pip install -r requirements.txt
```

## Preprocesamiento de documentos
Coloca tus archivos PDF en la carpeta `docs_osce/` y ejecuta:
```
python preprocesar.py
```

Esto generará una base vectorial `db_osce`.

## Ejecutar el servidor
```
uvicorn app.main:app --reload --port 8000
```

## Endpoint
POST `/buscar`
```json
{
  "query": "¿Cuáles son los supuestos de contratación directa?"
}
```

## Respuesta esperada
```json
{
  "contexto": "Texto relevante extraído de los documentos..."
}
```


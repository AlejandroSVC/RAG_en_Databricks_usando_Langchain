# RAG en Databricks usando Langchain

## Resumen:

Este código permite a un sistema RAG en Databricks procesar hasta 10 archivos PDF, fragmentando su contenido y almacenándolo en una base vectorial para enriquecer las respuestas generadas por el modelo de lenguaje.

## Sección 1: Importación de librerías necesarias
En esta sección, agregamos la capacidad de leer archivos PDF usando PyPDF y LangChain.
Esto permite al sistema procesar documentos en formato PDF, fragmentarlos y luego 
almacenarlos para la recuperación semántica.

```
from langchain.vectorstores import Chroma                  # Motor de almacenamiento y búsqueda de vectores
from langchain.embeddings import OpenAIEmbeddings          # Para crear embeddings de texto
from langchain.llms import OpenAI                          # Modelo de lenguaje de OpenAI
from langchain.chains import RetrievalQA                   # Cadena para RAG
from langchain.text_splitter import CharacterTextSplitter  # Particionador de texto
from langchain.document_loaders import PyPDFLoader         # Cargador de archivos PDF
import os                                                  # Para gestionar variables de entorno
import glob                                                # Para listar archivos PDF
```

## Sección 2: Preparación del entorno y configuración de las claves de API
Se configuran las variables de entorno necesarias para la autenticación y se 
especifica la ruta donde se encuentran los archivos PDF.

```
os.environ["OPENAI_API_KEY"] = "TU_CLAVE_OPENAI"           # Ingresar clave de API de OpenAI
pdf_dir = "/ruta/a/tus/pdfs"                               # Carpeta donde están los archivos PDF
```

## Sección 3: Carga y procesamiento de hasta 10 archivos PDF
En esta sección, buscamos hasta 10 archivos PDF en el directorio especificado,
los cargamos y fragmentamos su contenido para crear la base de conocimiento.

```
pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))[:10]  # Listar hasta 10 PDFs
splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)                # Fragmentar el texto
docs = []                                                  # Lista para los fragmentos

for pdf_path in pdf_files:                                 # Iterar sobre cada PDF
    loader = PyPDFLoader(pdf_path)                         # Cargar el PDF
    pages = loader.load()                                  # Extraer las páginas
    for page in pages:                                     # Iterar sobre las páginas
        docs.extend(splitter.split_documents([page]))      # Fragmentar y agregar
```

## Sección 4: Creación de la base de datos vectorial
Aquí se generan los embeddings para cada fragmento y se almacenan en una base vectorial usando Chroma, lo que permite realizar búsquedas semánticas sobre el contenido de los PDF.

```
embedding = OpenAIEmbeddings()                             # Inicializar los embeddings
vector_db = Chroma.from_documents(docs, embedding)         # Base de datos vectorial
```

## Sección 5: Configuración del modelo generativo y la cadena de RAG
Se inicializa el modelo LLM y se configura la cadena RetrievalQA, que realiza preguntas enriquecidas usando la información recuperada de los PDFs.

```
llm = OpenAI(temperature=0.2)                              # Inicializar el LLM
qa_chain = RetrievalQA.from_chain_type(                    # Crear la cadena RAG
    llm=llm,                                               # Modelo de lenguaje
    retriever=vector_db.as_retriever(),                    # Recuperador semántico
    return_source_documents=True                           # Devolver los docs fuente
)
```

## Sección 6: Ejecución de una consulta y obtención de la respuesta aumentada
Realiza una consulta de ejemplo y muestra la respuesta generada, junto con los fragmentos de documentos fuente utilizados.

```
pregunta = "Resume el contenido de los documentos PDF"      # Consulta de ejemplo
respuesta = qa_chain({"query": pregunta})                   # Ejecutar la consulta
print("Respuesta generada:", respuesta["result"])           # Mostrar la respuesta
for i, doc in enumerate(respuesta["source_documents"]):     # Recorrer los docs fuente
    print(f"Documento fuente {i+1}:",
        doc.page_content[:300], "...")                      # Muestra parte del contenido
```

'''

## Recomendaciones:

•  Asegurar que la ruta de los PDFs sea accesible desde el entorno Databricks.

•  Es posible ajustar el tamaño y solapamiento de los fragmentos según necesidades de contexto.

•  Para grandes volúmenes, considerar procesamiento distribuido y bases vectoriales persistentes.

•  Revisar las políticas de uso de la API de OpenAI para grandes cantidades de texto.

## Referencias:

•  LangChain: https://python.langchain.com/

•  PyPDFLoader:
  https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf

•  Chroma: https://docs.trychroma.com/

•  OpenAI API: https://platform.openai.com/docs/api-reference

•  Databricks: https://docs.databricks.com/

•  RAG (Retrieval Augmented Generation): https://arxiv.org/abs/2005.11401

'''

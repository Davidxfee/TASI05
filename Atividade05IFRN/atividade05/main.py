from bs4 import BeautifulSoup
import requests
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA

# URL 
url = "https://cienciahoje.org.br/artigo/a-era-da-inteligencia-artificial/"

print("Iniciando a requisição HTTP...")

# se a requisição HTTP foi bem-sucedida armazena o conteúdo HTML da página;
# se não, gera um erro com o código de status
response = requests.get(url)
print(f"Status da requisição: {response.status_code}")

if response.status_code == 200:
    html_content = response.text
    print("Conteúdo HTML carregado com sucesso.")
else:
    raise Exception(f"Falha ao carregar a página. Código HTTP: {response.status_code}")

# Extraindo o texto do HTML
print("Extraindo texto do HTML...")
soup = BeautifulSoup(html_content, "html.parser")
page_text = soup.get_text(separator="\n", strip=True)
print(f"Texto extraído, comprimento: {len(page_text)} caracteres.")

# Dividindo o texto em chunks
print("Dividindo o texto em chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Tamanho máximo dos chunks
    chunk_overlap=50,  # Sobreposição entre chunks
    length_function=len,
)
texts = text_splitter.split_text(page_text)  # Dividir o texto
print(f"Texto dividido em {len(texts)} chunks.")

# Ajuste na criação do banco de embeddings
print("Construindo o banco de embeddings com FAISS...")

embeddings = []
embedding_model = OllamaEmbeddings(model="hf.co/mixedbread-ai/mxbai-embed-large-v1:latest")

# Processar todos os chunk para teste
print("Processando todos chunk...")
chunk_embeddings = embedding_model.embed_documents(texts[:1])  # Processando apenas o primeiro chunk

# Criar uma lista de tuplas (texto, embedding)
embeddings.extend([(texts[i], chunk_embeddings[i]) for i in range(len(chunk_embeddings))])

# Adicionando um log para garantir que estamos com embeddings prontos
print(f"Total de embeddings processados: {len(embeddings)}")

# Criar o banco de dados com FAISS a partir dos embeddings gerados
print("Criando banco de embeddings com FAISS...")
try:
    # Fornecer os embeddings e o modelo para o FAISS
    db = FAISS.from_embeddings(embeddings, embedding_model)
    print("Banco de embeddings criado com sucesso.")
except Exception as e:
    print(f"Erro ao criar banco de embeddings: {str(e)}")
    raise
# Consulta com query para buscar chunks relacionados
query = "Quais foram os marcos históricos importantes no desenvolvimento da IA?"
print(f"Realizando consulta: '{query}'")
docs = db.similarity_search(query)
print(f"Consulta realizada, {len(docs)} documentos encontrados.")

# Configuração do modelo Ollama
model = OllamaLLM(model="llama3.2:latest")

# Criar o retriever e a chain de perguntas e respostas
print("Criando o retriever e a chain de perguntas e respostas...")
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff")
print("Retriever e chain criados com sucesso.")

# Usar a chain para responder perguntas
print("Obtendo resposta para a consulta...")
response = qa_chain.invoke(query)
print("QA Response:", response)

# Agentic LGPD

Agentic LGPD é um agente inteligente especializado em LGPD (Lei Geral de Proteção de Dados) e jurisprudência. Ele permite responder perguntas dos usuários com base em documentos legais e decisões judiciais, utilizando buscas híbridas e modelos de linguagem avançados.

---

## 🧩 Arquitetura do Agente

O pipeline do Agentic LGPD segue estas etapas:

### 1️⃣ Extração e pré-processamento de dados
- Documentos da LGPD e jurisprudência são divididos em chunks (trechos) com metadados.
- JSONs carregados: `lei_chunks_com_metadados_lei.json` e `lei_chunks_com_metadados_jurisprudencia.json`.
- Pré-carregamento de índices FAISS e BM25 para otimizar performance.
- Os dados foram extraídos de fontes públicas e retirados dados pessoais.

### 2️⃣ Embeddings e índices
- Modelo de embeddings: `sentence-transformers/all-mpnet-base-v2`.
- Índice FAISS para busca semântica.
- Índice BM25 para busca lexical baseada em palavras-chave.

### 3️⃣ Busca híbrida
Combina:
- **Busca semântica** (FAISS + embeddings)  
- **Busca lexical** (BM25)  
- Resultados combinados e sem duplicatas.

### 4️⃣ Workflow com LangGraph
O fluxo de estados do agente:

1. **check_question**: classifica a pergunta como `Lei`, `Jurisprudencia`, ambos ou fora do tema.  
2. **topic_router**: direciona para a recuperação correta de documentos.  
3. **retrieve_docs_***: busca os documentos relevantes.  
4. **generate**: gera resposta inicial com GPT-4o-mini.  
5. **improve_answer**: aprimora resposta considerando histórico e clareza.  
6. **END**: retorna resposta final.

### 5️⃣ Interface Gradio
- Chat interativo com histórico de até 5 turnos.  
- Limite de tamanho da pergunta para controle de tokens.  
- Pronto para testes em browser.

---

## ⚡ Tecnologias utilizadas
- Python 3.10+
- [LangGraph](https://github.com/langgraph/langgraph)
- [LangChain OpenAI](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [BM25Okapi](https://github.com/dorianbrown/rank_bm25)
- [Gradio](https://gradio.app/)

---

## 📦 Estrutura do projeto


import PyPDF2
import numpy as np
import re
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from .models import Document, DocumentChunk
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.schema  import retriever
from langchain.schema import Document as LCDocument
from huggingface_hub import InferenceClient
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class summarzie:
    def __init__(self):
        self._embediing_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.langchain_embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.key = HUGGING_FACE_KEY

        # Load once here
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.llm_pipeline = pipeline(
            task="summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,         # Controls output length
            truncation=True,            # Applies to input tokenization
        )
        self.hf_pipeline = HuggingFacePipeline(pipeline=self.llm_pipeline)

    def summarize_text(self, document: Document) -> str:
        try:
            chunks = DocumentChunk.objects.filter(document=document).order_by("chunk_index")
            if not chunks.exists():
                return "No document chunks found for this document"

            lc_documents = [
                LCDocument(page_content=chunk.content, metadata={"chunk_index": chunk.chunk_index})
                for chunk in chunks
            ]
            vector_store = FAISS.from_documents(lc_documents, self.langchain_embedding)
            retriever = vector_store.as_retriever()

            retriver_qa = RetrievalQA.from_chain_type(
                llm=self.hf_pipeline,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=False,
                output_key='text'
            )
            result = retriver_qa.invoke({"query": "Summarize the document"})
            raw_text = result["text"]

# Regex to extract after 'Answer:' or similar
            match = re.search(r"(?i)Answer\s*[:\-]\s*(.+)", raw_text)

            if match:
                answer = match.group(1).strip()
            else:
                answer = raw_text.strip()  # fallback: return whole text
        except Exception as e:
            return f"Failed to generate summary: {str(e)}"



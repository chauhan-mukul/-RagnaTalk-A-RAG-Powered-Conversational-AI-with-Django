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
import re
class RAGService:
    def __init__(self):
        self.model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.langchain_embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.key="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"
        

    def extract_text_from_pdf(self,pdf_file)->str:
        try:
            pdf_reader=PyPDF2.PdfReader(pdf_file)
            text=""
            for page in pdf_reader.pages:
                text+=page.extract_text()+'\n'
            return text
        except Exception as e:
            raise Exception(f"error extracting text from PDF :{str(e)}")
        
    def chuck_text(self,text:str,chunck_size:int=500,overlap:int=50)->List[str]:
        """splitting the text into chunks """
        text=re.sub(r'\s+',' ',text).strip()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk)+len(sentence)<chunck_size:
                current_chunk+=sentence+"."
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk=sentence+'.'
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def create_embeddings(self,chunks:List[str])->np.ndarray:
        embeddings=self.model.encode(chunks)
        return embeddings
    
    def process_document(self,document:Document)->Dict:
        try:
            text=self.extract_text_from_pdf(document.file)
            document.content=text
            document.save()
            chunks=self.chuck_text(text)
            embeddings=self.create_embeddings(chunks)
            DocumentChunk.objects.filter(document=document).delete()
            for i,(chunk,embedding) in enumerate(zip(chunks,embeddings)):
                DocumentChunk.objects.create(
                    document=document,
                    content=chunk,
                    chunk_index=i,
                    embedding=embedding.tolist()
                )
            return {
                'status':'success',
                'message': f'Document processed successfully. Created {len(chunks)} chunks.',
                'chunks_count': len(chunks)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error processing document: {str(e)}'
            }
        
   
    def generate_answer(self,query:str,document:Document)->str:
        try:
            chunks=DocumentChunk.objects.filter(document=document).order_by("chunk_index")
            if not chunks.exists():
                return "No document chunks found  for this documnet"
            lc_documents=[LCDocument(page_content=chunk.content,metadata={"chunk_index":chunk.chunk_index})
                         for chunk in chunks
            ]
            vector_store = FAISS.from_documents(lc_documents, self.langchain_embedding) 
            retriever = vector_store.as_retriever()
           

            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # You can replace with another local model path or Hugging Face ID
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            llm_pipeline = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
            hf_pipeline=HuggingFacePipeline(pipeline=llm_pipeline)
            retriver_qa = RetrievalQA.from_chain_type(
                    llm=hf_pipeline,
                    retriever=retriever,
                    chain_type="stuff",  # "stuff" = concatenate all context
                    return_source_documents=False,
                    output_key='text'
            )
            result = retriver_qa.invoke({"query": query})
            raw_text = result["text"]

# Regex to extract after 'Answer:' or similar
            match = re.search(r"(?i)answer\s*[:\-]\s*(.+)", raw_text)

            if match:
                answer = match.group(1).strip()
            else:
                answer = raw_text.strip()  # fallback: return whole text

            return(answer)
        except Exception as e:
            return f"Failed to generate answer: {str(e)}"

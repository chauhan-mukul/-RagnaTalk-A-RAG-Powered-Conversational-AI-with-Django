from django.shortcuts import render, get_object_or_404,HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.conf import settings
import json
import os

from .models import Document, Question,DocumentChunk,Summary
from .rag_engine import RAGService
from .summarize import summarzie

def home(request):
    """Render the main page"""
    return render(request, 'home.html')

@csrf_exempt
@require_http_methods(["POST"])
def upload_document(request):
    """Upload and process PDF document"""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file provided'}, status=400)
        file = request.FILES['file']
        title = request.POST.get('title', file.name)
        
        if not file.name.lower().endswith('.pdf'):
            return JsonResponse({'error': 'Only PDF files are allowed'}, status=400)
        
        document = Document.objects.create( #storing the document in documnet table
            title=title,
            file=file
            )
            
            
        rag_service = RAGService()# calling our rag service from  the backend
        result = rag_service.process_document(document) # processing the documents that is breaking in chunks and embedding
        # returning an json response
        if result['status'] == 'success':
            return JsonResponse({
                'document_id': str(document.id),
                'title': document.title,
                'message': result['message'],
                'chunks_count': result['chunks_count']
            })
        else:
            document.delete()
            return JsonResponse({'error': result['message']}, status=400)  
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def ask_question(request, document_id):
    try: 
        #Fetching the document
        document = get_object_or_404(Document, id=document_id)

        #Parse the incoming JSON
        payload = json.loads(request.body)
        question_text = payload.get("question", "").strip()
        if not question_text:
            return JsonResponse({"error": "Question is required"}, status=400)

        # running our ragservice that is our llm mnodel
        rag = RAGService()
        answer = rag.generate_answer(question_text, document)

        chunks_qs = DocumentChunk.objects.filter(document=document).order_by("chunk_index")
        context = "\n\n".join(c.content for c in chunks_qs)

        # storing the question with answer in database 
        question_obj = Question.objects.create(
            document=document,
            question_text=question_text,
            answer=answer,
            context=context[:2000]  # if you want to cap stored context
        )

        # Return the JSON payload your frontend expects
        return JsonResponse({
            "question_id": str(question_obj.id),
            "question": question_text,
            "answer": answer,
            "context": context[:1000],          # for display
            "similar_chunks": [c.content for c in chunks_qs][:3]  # e.g. show first 3 chunks
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    
@csrf_exempt
@require_http_methods(["DELETE"])
def delete_document(request, document_id):
    """Delete a document and all related data"""
    try:
        document = get_object_or_404(Document, id=document_id)
        
        # Delete the file from storage
        if document.file:
            if os.path.exists(document.file.path):
                os.remove(document.file.path)
        
        document.delete()
        return JsonResponse({'message': 'Document deleted successfully'})
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def document_questions(request, document_id):
    """Get all questions for a document"""
    document = get_object_or_404(Document, id=document_id)
    questions = Question.objects.filter(document=document).order_by('-created_at')
    
    questions_data = []
    for q in questions:
        questions_data.append({
            'id': str(q.id),
            'question_text': q.question_text,
            'answer': q.answer,
            'context': q.context[:200],  # Limit context
            'created_at': q.created_at.isoformat()
        })
    
    return JsonResponse({'questions': questions_data})

@require_http_methods(["GET"])
def list_documents(request):
    """List all uploaded documents (no changes needed here)"""
    documents = Document.objects.all().order_by("-created_at")
    docs = []
    for doc in documents:
        docs.append({
            "id": str(doc.id),
            "title": doc.title,
            "created_at": doc.created_at.isoformat(),
            "chunks_count": doc.chunks.count(),
        })
    return JsonResponse({"documents": docs})

@csrf_exempt
@require_http_methods(['POST'])
def summarizer(request,document_id):
    try: 
        #Fetching the document
        document = get_object_or_404(Document, id=document_id)

        #Parse the incoming JSON
        payload = json.loads(request.body)
        suma=summarzie()
        answer=suma.summarize_text(document)
        chunks_qs = DocumentChunk.objects.filter(document=document).order_by("chunk_index")
        context = "\n\n".join(c.content for c in chunks_qs)
        summarY_obj=Summary.objects.create(
            document=document,
            summary=answer,
            context=context[:2000]  # if you want to cap stored context
        )
        return JsonResponse({
            "question_id": str(summarY_obj.id),
            "summary": answer,
            "context": context[:1000]        # for display
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    
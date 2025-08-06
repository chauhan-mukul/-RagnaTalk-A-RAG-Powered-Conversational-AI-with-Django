from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('api/upload/', views.upload_document, name='upload_document'),
    path('api/documents/', views.list_documents, name='list_documents'),
    path('api/documents/<uuid:document_id>/ask/', views.ask_question, name='ask_question'),
    path('api/documents/<uuid:document_id>/questions/', views.document_questions, name='document_questions'),
    path('api/documents/<uuid:document_id>/delete/', views.delete_document, name='delete_document'),
    path('api/documents/<uuid:document_id>/summarize/',views.summarizer,name='summarizer'),
]

   
from django.contrib import admin
from .models import Document, DocumentChunk, Question ,Summary

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['title', 'created_at', 'updated_at']
    list_filter = ['created_at']
    search_fields = ['title', 'content']
    readonly_fields = ['id', 'created_at', 'updated_at']

@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ['document', 'chunk_index', 'content_preview']
    list_filter = ['document']
    search_fields = ['content']
    readonly_fields = ['id']
    
    def content_preview(self, obj):
        return obj.content[:100] + "..." if len(obj.content) > 100 else obj.content
    content_preview.short_description = 'Content Preview'

@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ['document', 'question_preview', 'created_at']
    list_filter = ['document', 'created_at']
    search_fields = ['question_text', 'answer']
    readonly_fields = ['id', 'created_at']
    
    def question_preview(self, obj):
        return obj.question_text[:50] + "..." if len(obj.question_text) > 50 else obj.question_text
    question_preview.short_description = 'Question'

admin.site.register(Summary)
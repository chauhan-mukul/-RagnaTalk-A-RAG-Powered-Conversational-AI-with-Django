from django.db import models
import uuid
# Create your models here.
class Document(models.Model):
    id=models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
    title=models.CharField(max_length=255)
    file=models.FileField(upload_to='documents/')
    content = models.TextField(blank=True, null=True)
    created_at=models.DateTimeField(auto_now_add=True)
    updated_at=models.DateField(auto_now=True)
    
    def __str__(self):
        return self.title

class DocumentChunk(models.Model):
    id=models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
    document=models.ForeignKey(Document,on_delete=models.CASCADE,related_name='chunks')
    content=models.TextField()
    chunk_index=models.IntegerField()
    embedding=models.JSONField(null=True,blank=True)
    class Meta:
        ordering =['chunk_index']
    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_index}"
    
class Question(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='questions')
    question_text = models.TextField()
    answer = models.TextField()
    context = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Q: {self.question_text[:50]}..."
    
class Summary(models.Model):
    id=models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
    document=models.ForeignKey(Document,on_delete=models.CASCADE,related_name='summarzier')
    summary=models.TextField()
    context=models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.document.title}"
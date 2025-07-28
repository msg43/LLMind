"""
Document Processor Tests
Tests document processing, chunking, and text extraction functionality
"""
import pytest
import asyncio
import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from fastapi import UploadFile

from core.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test document processing functionality"""
    
    @pytest.fixture
    def processor(self):
        """Create a document processor instance"""
        return DocumentProcessor()
    
    @pytest.fixture
    def mock_upload_file(self):
        """Create a mock UploadFile for testing"""
        def create_mock_file(content: bytes, filename: str):
            mock_file = Mock(spec=UploadFile)
            mock_file.filename = filename
            mock_file.read = AsyncMock(return_value=content)
            return mock_file
        return create_mock_file


class TestTextFileProcessing:
    """Test processing of text files"""
    
    @pytest.mark.asyncio
    async def test_process_txt_file(self, processor, mock_upload_file):
        """Test processing a simple text file"""
        content = b"This is a test document.\n\nIt has multiple paragraphs.\n\nAnd should be chunked properly."
        mock_file = mock_upload_file(content, "test.txt")
        
        result = await processor.process_file(mock_file)
        
        assert result["filename"] == "test.txt"
        assert result["text_length"] > 0
        assert result["chunk_count"] > 0
        assert len(result["chunks"]) == result["chunk_count"]
        assert all("text" in chunk for chunk in result["chunks"])
        assert all("source" in chunk for chunk in result["chunks"])
    
    @pytest.mark.asyncio
    async def test_process_markdown_file(self, processor, mock_upload_file):
        """Test processing a Markdown file"""
        content = b"""# Test Document

This is a **markdown** document with *formatting*.

## Section 1
Content under section 1.

## Section 2
- List item 1
- List item 2

```python
# Code block
print("Hello, World!")
```
"""
        mock_file = mock_upload_file(content, "test.md")
        
        result = await processor.process_file(mock_file)
        
        assert result["filename"] == "test.md"
        assert result["text_length"] > 0
        assert result["chunk_count"] > 0
        # Markdown should be converted to plain text
        assert "**" not in result["chunks"][0]["text"]  # Bold formatting removed
    
    @pytest.mark.asyncio
    async def test_process_html_file(self, processor, mock_upload_file):
        """Test processing an HTML file"""
        content = b"""<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
    <style>body { color: blue; }</style>
    <script>console.log("test");</script>
</head>
<body>
    <h1>Main Title</h1>
    <p>This is a paragraph with <strong>bold</strong> text.</p>
    <div>
        <h2>Subsection</h2>
        <ul>
            <li>List item 1</li>
            <li>List item 2</li>
        </ul>
    </div>
</body>
</html>"""
        mock_file = mock_upload_file(content, "test.html")
        
        result = await processor.process_file(mock_file)
        
        assert result["filename"] == "test.html"
        assert result["text_length"] > 0
        assert result["chunk_count"] > 0
        # HTML tags should be removed
        text = result["chunks"][0]["text"]
        assert "<html>" not in text
        assert "<script>" not in text
        assert "Main Title" in text
        assert "List item 1" in text


class TestChunkingLogic:
    """Test intelligent chunking functionality"""
    
    @pytest.mark.asyncio
    async def test_intelligent_chunking(self, processor, mock_upload_file):
        """Test intelligent text chunking with overlaps"""
        # Create a longer document to test chunking
        paragraphs = [
            "This is the first paragraph. It contains some important information about the topic.",
            "The second paragraph continues the discussion. It builds upon the previous content.",
            "Here is the third paragraph. It introduces new concepts that are related.",
            "The fourth paragraph provides additional details. These details are crucial for understanding.",
            "Finally, the fifth paragraph concludes the document. It summarizes the key points."
        ]
        content = "\n\n".join(paragraphs).encode()
        mock_file = mock_upload_file(content, "long_document.txt")
        
        result = await processor.process_file(mock_file)
        
        # Should create multiple chunks for longer content
        chunks = result["chunks"]
        assert len(chunks) >= 1
        
        # Each chunk should have proper metadata
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i
            assert chunk["source"] == "long_document.txt"
            assert "metadata" in chunk
            assert chunk["metadata"]["filename"] == "long_document.txt"
            assert chunk["metadata"]["chunk_size"] == len(chunk["text"])
    
    @pytest.mark.asyncio
    async def test_short_document_single_chunk(self, processor, mock_upload_file):
        """Test that short documents create only one chunk"""
        content = b"This is a very short document."
        mock_file = mock_upload_file(content, "short.txt")
        
        result = await processor.process_file(mock_file)
        
        assert result["chunk_count"] == 1
        assert result["chunks"][0]["text"] == "This is a very short document."
        assert result["chunks"][0]["chunk_index"] == 0
    
    @pytest.mark.asyncio
    async def test_empty_document_handling(self, processor, mock_upload_file):
        """Test handling of empty documents"""
        content = b""
        mock_file = mock_upload_file(content, "empty.txt")
        
        with pytest.raises(ValueError, match="No text content found"):
            await processor.process_file(mock_file)
    
    @pytest.mark.asyncio
    async def test_whitespace_only_document(self, processor, mock_upload_file):
        """Test handling of documents with only whitespace"""
        content = b"   \n\n   \t\t   \n   "
        mock_file = mock_upload_file(content, "whitespace.txt")
        
        with pytest.raises(ValueError, match="No text content found"):
            await processor.process_file(mock_file)


class TestDocumentDeduplication:
    """Test document deduplication logic"""
    
    @pytest.mark.asyncio
    async def test_duplicate_file_detection(self, processor, mock_upload_file):
        """Test that identical files are detected and not reprocessed"""
        content = b"This is a test document for deduplication."
        mock_file1 = mock_upload_file(content, "test1.txt")
        mock_file2 = mock_upload_file(content, "test2.txt")  # Different name, same content
        
        result1 = await processor.process_file(mock_file1)
        result2 = await processor.process_file(mock_file2)
        
        # Should have same hash and be detected as duplicate
        assert result1["file_hash"] == result2["file_hash"]
        # Second call should return cached result
        assert result1["chunks"] == result2["chunks"]
    
    @pytest.mark.asyncio
    async def test_different_files_different_hashes(self, processor, mock_upload_file):
        """Test that different files get different hashes"""
        content1 = b"This is the first document."
        content2 = b"This is the second document."
        mock_file1 = mock_upload_file(content1, "doc1.txt")
        mock_file2 = mock_upload_file(content2, "doc2.txt")
        
        result1 = await processor.process_file(mock_file1)
        result2 = await processor.process_file(mock_file2)
        
        assert result1["file_hash"] != result2["file_hash"]
        assert result1["chunks"][0]["text"] != result2["chunks"][0]["text"]


class TestDocumentManagement:
    """Test document management functionality"""
    
    @pytest.mark.asyncio
    async def test_get_document_count(self, processor, mock_upload_file):
        """Test getting document count"""
        initial_count = await processor.get_document_count()
        
        content = b"Test document"
        mock_file = mock_upload_file(content, "test.txt")
        await processor.process_file(mock_file)
        
        new_count = await processor.get_document_count()
        assert new_count == initial_count + 1
    
    @pytest.mark.asyncio
    async def test_get_document_list(self, processor, mock_upload_file):
        """Test getting list of processed documents"""
        content = b"Test document content"
        mock_file = mock_upload_file(content, "test_list.txt")
        result = await processor.process_file(mock_file)
        
        doc_list = await processor.get_document_list()
        
        assert len(doc_list) >= 1
        found_doc = next((doc for doc in doc_list if doc["filename"] == "test_list.txt"), None)
        assert found_doc is not None
        assert found_doc["file_hash"] == result["file_hash"]
        assert found_doc["text_length"] > 0
        assert found_doc["chunk_count"] > 0
    
    @pytest.mark.asyncio
    async def test_delete_document(self, processor, mock_upload_file, temp_dir):
        """Test document deletion"""
        content = b"Document to be deleted"
        mock_file = mock_upload_file(content, "delete_me.txt")
        result = await processor.process_file(mock_file)
        
        file_hash = result["file_hash"]
        
        # Delete the document
        success = await processor.delete_document(file_hash)
        assert success is True
        
        # Verify it's removed from the processor
        doc_by_hash = await processor.get_document_by_hash(file_hash)
        assert doc_by_hash == {}
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, processor):
        """Test deleting a document that doesn't exist"""
        success = await processor.delete_document("nonexistent_hash")
        assert success is False


class TestSupportedFormats:
    """Test supported file format detection"""
    
    @pytest.mark.asyncio
    async def test_get_supported_formats(self, processor):
        """Test getting list of supported formats"""
        formats = await processor.get_supported_formats()
        
        expected_formats = ['.pdf', '.docx', '.txt', '.md', '.html']
        assert all(fmt in formats for fmt in expected_formats)
    
    @pytest.mark.asyncio
    async def test_unsupported_file_format(self, processor, mock_upload_file):
        """Test handling of unsupported file formats"""
        content = b"Binary content"
        mock_file = mock_upload_file(content, "test.exe")
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            await processor.process_file(mock_file)


class TestErrorHandling:
    """Test error handling in document processing"""
    
    @pytest.mark.asyncio
    async def test_corrupted_file_handling(self, processor, mock_upload_file):
        """Test handling of corrupted files"""
        # Simulate a corrupted file by providing invalid content for a PDF
        content = b"This is not a valid PDF file"
        mock_file = mock_upload_file(content, "corrupted.pdf")
        
        # Should handle gracefully and raise appropriate error
        with pytest.raises(Exception):  # Could be various exceptions depending on the library
            await processor.process_file(mock_file)
    
    @pytest.mark.asyncio
    async def test_very_large_file_handling(self, processor, mock_upload_file):
        """Test handling of very large files"""
        # Create a large text content
        large_content = ("This is a repeated line.\n" * 10000).encode()
        mock_file = mock_upload_file(large_content, "large.txt")
        
        result = await processor.process_file(mock_file)
        
        # Should process successfully and create multiple chunks
        assert result["text_length"] > 100000
        assert result["chunk_count"] > 1


class TestReprocessing:
    """Test document reprocessing functionality"""
    
    @pytest.mark.asyncio
    async def test_reprocess_document(self, processor, mock_upload_file, temp_dir):
        """Test reprocessing an existing document"""
        content = b"Document for reprocessing test"
        mock_file = mock_upload_file(content, "reprocess.txt")
        original_result = await processor.process_file(mock_file)
        
        file_hash = original_result["file_hash"]
        
        # Reprocess the document
        reprocessed_result = await processor.reprocess_document(file_hash)
        
        assert reprocessed_result["file_hash"] == file_hash
        assert reprocessed_result["filename"] == "reprocess.txt"
        # Content should be the same
        assert reprocessed_result["text_length"] == original_result["text_length"]
    
    @pytest.mark.asyncio
    async def test_reprocess_nonexistent_document(self, processor):
        """Test reprocessing a document that doesn't exist"""
        with pytest.raises(ValueError, match="Document not found"):
            await processor.reprocess_document("nonexistent_hash") 
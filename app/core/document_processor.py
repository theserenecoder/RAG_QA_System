"""Document Processing module for loading and chunking documents"""

import tempfile
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.logger import get_logger
from app.config import get_settings


logger = get_logger(__name__)

class DocumentProcessor:
    """Processing document for RAG Pipeline"""
    
    SUPPORTED_EXTENSIONS = {'.pdf','.txt','.csv'}
    
    def __init__(
        self,
        chunk_size:int | None = None, 
        chunk_overlap:int | None=None
        ):
        """Initializing Document Processor"""
        
        settings = get_settings()
        
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            seperators = ["\n\n","\n",".",","]
        )
        
        logger.info(
            f"DocumentProcessor Initialized with chunk size = {self.chunk_size},"
            f"chunk overlap = {self.chunk_overlap}"
        )
        
    def load_pdf(self,file_path: str | Path) -> list[Document]:
        """Load a PDF File
        
        Args:
            file_path : Path to PDF file
            
        Returns:
            List of document objects
        """
        
        file_path = Path(file_path)
        logger.info(f"Loading PDF: {file_path.name}")
        
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        
        logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
        return documents
    
    def load_text(self, file_path:str | Path) -> list[Document]:
        """Load a text file

        Args:
            file_path : Path to text file

        Returns:
            list[Document]: List of document objects
        """
        
        file_path = Path(file_path)
        logger.info(f"Loading text file : {file_path.name}")
        
        loader = TextLoader(str(file_path))
        documents = loader.load()
        
        logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
        return documents
    
    def load_csv(self, file_path : str | Path) -> list[Document]:
        """Load a csv file

        Args:
            file_path : Path to csv file

        Returns:
            list[Document]: List of document objects
        """
        
        file_path = Path(file_path)
        logger.info(f"Loading csv file : {file_path.name}")
        
        loader = CSVLoader(str(file_path))
        documents = loader.load()
        
        logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
        return documents
    
    def load_file(self, file_path: str | Path) -> list[Document]:
        """Load a file based on its extension
        
        Args:
            file_path : Path of the file
        
        Return:
            List of Document Objects
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {extension}."
                f"Supported extensions: {self.SUPPORTED_EXTENSIONS}"
            )
        
        loader = {
            '.pdf': self.load_pdf,
            '.txt' : self.load_text,
            '.csv': self.load_csv
        }
        
        return loader[extension](file_path)
    
    def load_from_upload(
        self,
        file: BinaryIO,
        filename: str
    ) -> list[Document]:
        """Load document from uploaded file
        Args:
            file : File-like object
            filename : Original Filename
        Return:
            List of Document Objects
        """
        extension = Path(filename).suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {extension}."
                f"Supported extensions: {self.SUPPORTED_EXTENSIONS}"
            )
        
        ## saving in temorary file for saving
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
            
        try:
            documents = self.load_file(tmp_path)
            
            ## update metadata with original file
            for doc in documents:
                doc.metadata["source"] = filename
            
            return documents
        
        finally:
            ## clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
            
    def split_documents(self,documents: list[Document]) -> list[Document]:
        """Split Documents into chunks
        
        Args:
            documents: List of document objects
        Return:
            List of chunked document objects
        """        
        logger.info(f"Splitting {len(documents)} documents into chunk")
        
        chunks = self.text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def process_file(self,file_path: str | Path) -> list[Document]:
        """Load and split file in one process
        Args:
            file_path : Path of the file
        Returns:
            List of document objects
        """
        documents = self.load_file(file_path)
        chunked_documents = self.split_documents(documents)
        return chunked_documents
    
    def process_upload(
        self,
        file: BinaryIO,
        filename: str
    ) -> list[Document]:
        """Load and split uploaded file

        Args:
            file (BinaryIO): File-like object
            filename (str): Original filename

        Returns:
            List of document objects
        """
        documents = self.load_from_upload(file,filename)
        chunked_documents = self.split_documents(documents)
        return chunked_documents

   
if __name__ == "__main__":    
    obj = DocumentProcessor()
    
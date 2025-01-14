import os.path

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

path = os.path.curdir
loader  = Docx2txtLoader(os.path.join(path, '..' ,'data_sets', 'Training Notes.docx'))

text_splitter = CharacterTextSplitter(

    separators= "\n\n",
    chunk_size=100,  # Size of each chunk in characters
    chunk_overlap=10,  # Overlap between consecutive chunks
    length_function=len,  # Function to compute the length of the text
    add_start_index=True,  # Flag to add start index to each chunk,
    #is_separator_regex=False,
)

chunks = text_splitter.split_documents(documents=loader.load())

pass
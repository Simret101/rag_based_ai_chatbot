import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import os

def process_drugs_dataset(input_csv="cleaned_drugs_dataset.csv", output_dir="vector_store/"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(input_csv)
    
    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Prepare documents
    documents = []
    for _, row in df.iterrows():
        drug_name = row['drug_name']
        side_effects = row['side_effects']
        pregnancy_category = row['pregnancy_category']
        
        # Create document content
        content = f"Drug: {drug_name}\nSide Effects: {side_effects}\nPregnancy Category: {pregnancy_category}"
        
        # Create Document object
        document = Document(
            page_content=content,
            metadata={
                "drug_name": drug_name,
                "pregnancy_category": pregnancy_category
            }
        )
        
        # Add to documents list
        documents.append(document)
    
    # Create vector store
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(output_dir)
    
    print(f"Vector store created and saved to {output_dir}")

if __name__ == "__main__":
    process_drugs_dataset()

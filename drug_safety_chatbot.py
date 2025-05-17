from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
import google.generativeai as genai
import os
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI(title="Drug Safety Chatbot API")

# Request model for validation
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    similarity_score: int
    drug_name: str
    is_pregnancy_query: bool

# List of common greetings
GREETINGS = [
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening",
    "howdy", "yo", "what's up", "how are you", "how's it going"
]

# Function to detect greetings
def is_greeting(text):
    # Check if the text is a greeting using fuzzy matching
    for greeting in GREETINGS:
        if fuzz.ratio(text.lower(), greeting) > 80:  # 80% similarity threshold
            return True
    return False

# Function to handle greetings
def handle_greeting():
    return "Hi there! I'm here to help with drug safety information. How can I assist you today?"

# Load environment variables
load_dotenv()

# Set up API key
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize models
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=api_key)

# Load vector store
def load_vector_store(vector_store_dir="vector_store/"):
    if not os.path.exists(vector_store_dir):
        os.makedirs(vector_store_dir)
    vector_store = FAISS.load_local(vector_store_dir, embedding_model, allow_dangerous_deserialization=True)
    return vector_store

# Create RAG chain
def create_rag_chain():
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return rag_chain

# Pregnancy category explanations
def get_pregnancy_explanation(category, drug_name):
    detailed_explanations = {
        "A": f"{drug_name} is in pregnancy category A. This means that adequate and well-controlled studies have failed to demonstrate a risk to the fetus in the first trimester of pregnancy, and there is no evidence of risk in later trimesters. It is considered safe during pregnancy.",
        "B": f"{drug_name} is in pregnancy category B. This means that animal reproduction studies have failed to demonstrate a risk to the fetus and there are no adequate and well-controlled studies in pregnant women OR animal studies have shown an adverse effect, but adequate and well-controlled studies in pregnant women have failed to demonstrate a risk to the fetus in any trimester.",
        "C": f"{drug_name} is in pregnancy category C. This means that animal studies have shown an adverse effect on the fetus and there are no adequate and well-controlled studies in humans, but potential benefits may warrant use of the drug in pregnant women despite potential risks.",
        "D": f"{drug_name} is in pregnancy category D. This means that there is positive evidence of human fetal risk based on adverse reaction data from investigational or marketing experience or studies in humans, but potential benefits may warrant use of the drug in pregnant women despite potential risks.",
        "X": f"{drug_name} is in pregnancy category X. This means that studies in animals or humans have demonstrated fetal abnormalities and/or there is positive evidence of human fetal risk based on adverse reaction data from investigational or marketing experience, and the risks involved in use of the drug in pregnant women clearly outweigh potential benefits.",
        "N": f"{drug_name} has not been classified for pregnancy safety. This means there is insufficient information to determine its safety during pregnancy. Consult a healthcare provider for personalized advice."
    }
    
    # Add a general note about consulting healthcare providers
    general_note = "\n\nNote: Always consult with a healthcare provider before taking any medication during pregnancy, as individual circumstances may vary."
    
    return detailed_explanations.get(category.upper(), detailed_explanations["N"]) + general_note

# Function to get similar drugs
def get_similar_drugs(query, df):
    # Get all drug names from the dataset
    drug_names = df['drug_name'].tolist()
    
    # Add common drug synonyms
    drug_synonyms = {
        'paracetamol': ['panadol', 'acetaminophen', 'tylenol'],
        'ibuprofen': ['brufen', 'nurofen'],
        'aspirin': ['acetylsalicylic acid'],
        'naproxen': ['aleve', 'naprosyn']
    }
    
    # Add synonyms to the search list
    for drug, synonyms in drug_synonyms.items():
        if drug not in drug_names:
            drug_names.append(drug)
        for synonym in synonyms:
            if synonym not in drug_names:
                drug_names.append(synonym)
    
    # Find the top 5 most similar drug names
    similar_drugs = process.extract(query, drug_names, limit=5)
    
    # If no drugs were found, return an empty list
    if not similar_drugs:
        return []
        
    return similar_drugs

# Function to check if query is about pregnancy
def is_pregnancy_query(query):
    # List of pregnancy-related keywords and variations
    pregnancy_keywords = [
        "during pregnancy",
        "pregnancy category",
        "safe during pregnancy",
        "pregnancy safety",
        "while pregnant",
        "in pregnancy",
        "for pregnant",
        "when pregnant",
        "if pregnant",
        "pregnancy use",
        "pregnancy risk",
        "pregnancy safety",
        "safe in pregnancy",
        "safe to use",
        "can i take",
        "should i take",
        "is it safe"
    ]
    
    # Check if any keyword matches the query with fuzzy matching
    for keyword in pregnancy_keywords:
        if fuzz.partial_ratio(keyword, query.lower()) > 70:  # 70% similarity threshold
            return True
    return False

# Function to extract drug name from natural language questions
def normalize_query(query):
    # Normalize text (keep it simple)
    query = query.lower().strip()
    
    # Common drug name synonyms
    drug_synonyms = {
        'panadol': 'paracetamol',
        'acetaminophen': 'paracetamol',
        'tylenol': 'paracetamol',
        'ibuprofen': ['brufen', 'nurofen'],
        'aspirin': ['acetylsalicylic acid'],
        'naproxen': ['aleve', 'naprosyn']
    }
    
    # Replace drug synonyms
    for drug, synonyms in drug_synonyms.items():
        if drug in query:
            query = query.replace(drug, 'paracetamol')  # Use paracetamol as a placeholder
            continue
        if isinstance(synonyms, list):
            for synonym in synonyms:
                if synonym in query:
                    query = query.replace(synonym, drug)
                    break
        else:
            if synonyms in query:
                query = query.replace(synonyms, drug)
    
    # Replace common typos
    query = query.replace('sideeffeects', 'side effects')
    query = query.replace('sideeffects', 'side effects')
    query = query.replace('effects', 'side effects')
    query = query.replace('adverse', 'side effects')
    query = query.replace('risks', 'side effects')
    
    # Keep the query as is, but make sure it's not empty
    if not query.strip():
        return ""
    
    return query

# Chatbot function
def drug_safety_chat(query):
    try:
        # Normalize query
        normalized_query = normalize_query(query)
        
        # Load dataset to get drug names for fuzzy search
        df = pd.read_csv("cleaned_drugs_dataset (1).csv")
        
        # Check for similar drug names
        similar_drugs = get_similar_drugs(normalized_query, df)
        
        # If no good matches found, suggest similar drugs
        if similar_drugs[0][1] < 60:  # If best match is less than 60% similar
            # Format suggestions with similarity scores
            suggestions = []
            for drug, score in similar_drugs:
                suggestions.append(f"{drug} ({score}% similar)")
            return f"I couldn't find information about '{query}'. Here are some similar drugs that might help:\n\n{', '.join(suggestions)}"
        
        # If we found a good match, use it for the query
        most_similar_drug = similar_drugs[0][0]
        similarity_score = similar_drugs[0][1]
        
        # Get the drug information from the dataset
        drug_info = df[df['drug_name'] == most_similar_drug].iloc[0]
        
        # Create a natural language response
        if "during pregnancy" in normalized_query:
            response = f"The provided text states that {most_similar_drug} is in Pregnancy Category {drug_info['pregnancy_category']}. {get_pregnancy_explanation(drug_info['pregnancy_category'], most_similar_drug)}"
        else:
            response = f"{most_similar_drug} may cause the following side effects: {drug_info['side_effects']}"
        
        # Add similarity information if it's not an exact match
        if similarity_score < 100:
            response = f"Found information about {most_similar_drug} ({similarity_score}% similar to your search):\n\n{response}"
        
        # Get sources using RAG chain
        rag_chain = create_rag_chain()
        result = rag_chain.invoke({"query": normalized_query})
        source_docs = result.get("source_documents", [])
        
        # Format sources in a cleaner way
        if source_docs:
            source_text = "\nSources:\n"
            for i, doc in enumerate(source_docs, 1):
                source_text += f"\n{i}. {doc.metadata.get('drug_name', 'Unknown')} (Pregnancy Category: {doc.metadata.get('pregnancy_category', 'N/A')})"
            response += source_text
        
        return response
    except Exception as e:
        return str(e)

# Function to handle chat responses
def handle_chat(query):
    try:
        # First check if it's a greeting
        if is_greeting(query):
            return ChatResponse(
                response=handle_greeting(),
                similarity_score=100,
                drug_name="",
                is_pregnancy_query=False
            )
            
        # Load dataset to get drug names for fuzzy search
        df = pd.read_csv("cleaned_drugs_dataset (1).csv")
        
        # Extract drug name from query
        query_lower = query.lower()
        
        # Check if query contains "during pregnancy"
        is_pregnancy_query = "during pregnancy" in query_lower
        
        # Extract drug name by removing common query patterns
        drug_name = query_lower
        drug_patterns = [
            "can i use ",
            "can i take ",
            "is it safe to use ",
            "is it safe to take ",
            "during pregnancy",
            "while pregnant",
            "in pregnancy"
        ]
        
        for pattern in drug_patterns:
            if pattern in drug_name:
                drug_name = drug_name.replace(pattern, "").strip()
        
        # Try to find exact match first
        exact_match = df[df['drug_name'].str.lower() == drug_name].to_dict('records')
        if exact_match:
            drug_info = exact_match[0]
        else:
            # Try fuzzy matching on just the drug name part
            drug_names = df['drug_name'].str.lower().tolist()
            similar_drugs = process.extract(drug_name, drug_names, limit=1)
            
            if similar_drugs and similar_drugs[0][1] >= 60:  # Require at least 60% similarity
                drug_name = similar_drugs[0][0]
                drug_info = df[df['drug_name'].str.lower() == drug_name.lower()].iloc[0].to_dict()
            else:
                # Format suggestions with similarity scores
                suggestions = []
                for drug, score in similar_drugs:
                    if score >= 60:  # Only show suggestions with good similarity
                        suggestions.append(f"{drug}")
                if suggestions:
                    return ChatResponse(
                        response=f"I couldn't find information about '{drug_name}'. Here are some similar drugs that might help:\n\n{', '.join(suggestions)}",
                        similarity_score=0,
                        drug_name=drug_name,
                        is_pregnancy_query=is_pregnancy_query
                    )
                else:
                    return ChatResponse(
                        response=f"I couldn't find information about '{drug_name}'. Please try a different drug name.",
                        similarity_score=0,
                        drug_name=drug_name,
                        is_pregnancy_query=is_pregnancy_query
                    )
        
        # Create a natural language response
        if is_pregnancy_query:
            response = f"The provided text states that {drug_name} is in Pregnancy Category {drug_info['pregnancy_category']}. {get_pregnancy_explanation(drug_info['pregnancy_category'], drug_name)}"
        else:
            response = f"{drug_name} may cause the following side effects: {drug_info['side_effects']}"
        
        # Add similarity information if we used fuzzy matching
        if drug_name.lower() != query_lower:
            similarity_score = fuzz.ratio(drug_name.lower(), query_lower)
            response = f"Found information about {drug_name} ({similarity_score}% similar to your search query):\n\n{response}"
        else:
            similarity_score = 100
        
        return ChatResponse(
            response=response,
            similarity_score=similarity_score,
            drug_name=drug_name,
            is_pregnancy_query=is_pregnancy_query
        )
    except Exception as e:
        print(f"Error in handle_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add FastAPI endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    return handle_chat(request.query)

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )

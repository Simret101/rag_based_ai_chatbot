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

# Load environment variables
load_dotenv()

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

# Function to get pregnancy category explanation
def get_pregnancy_explanation(category, drug_name):
    explanations = {
        'A': f"{drug_name} is in Pregnancy Category A. This means that adequate and well-controlled studies have failed to demonstrate a risk to the fetus in the first trimester of pregnancy, and there is no evidence of risk in later trimesters.",
        'B': f"{drug_name} is in Pregnancy Category B. This means that animal reproduction studies have failed to demonstrate a risk to the fetus and there are no adequate and well-controlled studies in pregnant women.",
        'C': f"{drug_name} is in Pregnancy Category C. This means that animal studies have shown an adverse effect on the fetus and there are no adequate and well-controlled studies in humans, but potential benefits may warrant use of the drug in pregnant women despite potential risks.",
        'D': f"{drug_name} is in Pregnancy Category D. This means that there is positive evidence of human fetal risk based on adverse reaction data from investigational or marketing experience or studies in humans, but potential benefits may warrant use of the drug in pregnant women despite potential risks.",
        'X': f"{drug_name} is in Pregnancy Category X. This means that studies in animals or humans have demonstrated fetal abnormalities and/or there is positive evidence of fetal risk based on human experience, and the risk of the use of the drug in pregnant women clearly outweighs any possible benefit.",
        'N': f"{drug_name} is in Pregnancy Category N. This means that the drug's effect on pregnancy is not known."
    }
    return explanations.get(category.upper(), f"The pregnancy category for {drug_name} is {category}")

# Function to check if text is a greeting
def is_greeting(text):
    return any(greeting in text.lower() for greeting in GREETINGS)

# Function to handle greetings
def handle_greeting():
    return "Hello! I'm here to help with drug safety information. You can ask about side effects and pregnancy safety categories."

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
        df = pd.read_csv("cleaned_drugs_dataset.csv")
        
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

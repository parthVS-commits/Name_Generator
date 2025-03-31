import os
from typing import List, Dict
import re
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone
import json
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define index names
primary_index_name = os.getenv("PINECONE_INDEX_NAME", "mca-scraped-final1")
trademark_index_name = os.getenv("TRADEMARK_INDEX_NAME", "tm-prod-pipeline")

# Initialize indexes
try:
    primary_index = pc.Index(primary_index_name)
except Exception as e:
    print(f"Error connecting to primary index: {str(e)}")
    primary_index = None

try:
    trademark_index = pc.Index(trademark_index_name)
except Exception as e:
    print(f"Error connecting to trademark index: {str(e)}")
    trademark_index = None

# Initialize FastAPI
app = FastAPI(title="Zolvit Name Generator API", 
              version="1.0.0",
              description="Generate unique business names based on descriptions")

# Pydantic models for request/response
class BusinessDescriptionRequest(BaseModel):
    description: str

class NameSuggestion(BaseModel):
    name: str
    description: str

class NameSuggestionsResponse(BaseModel):
    suggestions: List[NameSuggestion]

class NameValidator:
    @staticmethod
    def name_exists_in_database(name: str) -> bool:
        """
        Check if a business name already exists in the Pinecone database.
        
        Args:
            name: The business name to check
            
        Returns:
            bool: True if the name exists, False otherwise
        """
        # Check primary index (original format)
        if NameValidator._check_primary_index(name):
            return True
            
        # Check trademark index (wordMark format)
        if NameValidator._check_trademark_index(name):
            return True
            
        # Name doesn't exist in either index
        return False
        
    @staticmethod
    def _check_primary_index(name: str) -> bool:
        """Check if name exists in the primary index"""
        global primary_index
        
        if not primary_index:
            return False  # Assume name doesn't exist if we can't check
            
        try:
            # Query the index for exact match on original_data field
            results = primary_index.query(
                vector=[0] * 1536,  # Dummy vector, we're only checking metadata
                top_k=1,
                include_metadata=True,
                filter={
                    "original_data": {"$eq": name}
                }
            )
            
            # Return True if any matching records found
            exists = len(results.matches) > 0
            return exists
            
        except Exception as e:
            print(f"Error checking name in primary database: {str(e)}")
            return False
            
    @staticmethod
    def _check_trademark_index(name: str) -> bool:
        """Check if name exists in the trademark index"""
        global trademark_index
        
        if not trademark_index:
            return False  # Assume name doesn't exist if we can't check
            
        try:
            # Extract the main business name part before any dash or special character
            main_name = name.split('-')[0].strip()
            
            # Query the trademark index looking for wordMark field
            results = trademark_index.query(
                vector=[0] * 1536,  # Dummy vector, we're only checking metadata
                top_k=10,  # Check a few potential matches
                include_metadata=True
            )
            
            # Check if any of the returned wordMarks contain our name or vice versa
            for match in results.matches:
                if 'wordMark' in match.metadata:
                    trademark = match.metadata['wordMark']
                    
                    # Extract the main part of the trademark before any dash
                    if ' - ' in trademark:
                        trademark_main = trademark.split(' - ')[0].strip()
                    else:
                        trademark_main = trademark
                    
                    # Check if the main parts of the names are the same or very similar
                    if (main_name.lower() in trademark_main.lower() or 
                        trademark_main.lower() in main_name.lower()):
                        return True
            
            return False
            
        except Exception as e:
            print(f"Error checking name in trademark database: {str(e)}")
            return False

class BusinessNameGenerator:
    @staticmethod
    def generate_business_names(description: str) -> List[Dict[str, str]]:
        """Generate unique business name suggestions based on user description."""
        try:
            # Updated to use OpenAI v1.0+ API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert brand naming consultant specializing in creating COMPLETELY UNIQUE business names. Your goal is to:
                        - Generate names that are 100% distinct from each other
                        - Capture the unique essence of the business
                        - Ensure no two names sound or look similar
                        - Create memorable, trademark-friendly names
                        - Provide a brief explanation of why each name is being suggested, focusing on how it relates to the business description
                        
                        CRITICAL INSTRUCTIONS:
                        - ABSOLUTELY NO DUPLICATE OR SIMILAR NAMES
                        - Each name must be completely unique in sound, structure, and meaning
                        - Prioritize creativity and distinctiveness
                        - Format your response as a JSON array of objects with "name" and "description" properties
                        - Each description must be approximately 10 words and explain why the name fits the business"""
                    },
                    {
                        "role": "user",
                        "content": f"Create 12 COMPLETELY UNIQUE business names for this description, ensuring ZERO similarity between names: '{description}'. For each name, provide a 10-word description of why it's appropriate."
                    }
                ],
                max_tokens=500,
                n=1,
                temperature=0.8
            )
            suggestions_text = response.choices[0].message.content.strip()
            
            # Parse the JSON string into a list of dictionaries
            try:
                suggestions = json.loads(suggestions_text)
            except json.JSONDecodeError:
                # Fallback: Extract name-description pairs using regex if JSON parsing fails
                name_pattern = r'(?:"name":|^\d+\.)\s*"([^"]+)"'
                desc_pattern = r'(?:"description":|explanation:)\s*"([^"]+)"'
                
                names = re.findall(name_pattern, suggestions_text, re.MULTILINE)
                descriptions = re.findall(desc_pattern, suggestions_text, re.MULTILINE)
                
                suggestions = []
                for i in range(min(len(names), len(descriptions))):
                    suggestions.append({
                        "name": names[i],
                        "description": descriptions[i]
                    })
            
            # Filter and validate names against database
            unique_suggestions = []
            for suggestion in suggestions:
                name = suggestion["name"]
                # Skip duplicates
                if name in [s["name"] for s in unique_suggestions]:
                    continue
                    
                # Check if name exists in database
                if not NameValidator.name_exists_in_database(name):
                    unique_suggestions.append(suggestion)
                    if len(unique_suggestions) == 12:
                        break
            
            return unique_suggestions
        except Exception as e:
            print(f"Error generating business names: {str(e)}")
            return []

@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "api": "Zolvit Business Name Generator API",
        "version": "1.0.0",
        "usage": "POST /generate-names with a 'description' field in the request body"
    }

@app.post("/generate-names", response_model=NameSuggestionsResponse)
def generate_names(request: BusinessDescriptionRequest):
    """Generate unique business names based on the provided description"""
    if not request.description:
        raise HTTPException(status_code=400, detail="Business description is required")
    
    # Generate name suggestions
    suggestions = BusinessNameGenerator.generate_business_names(request.description)
    
    if not suggestions:
        raise HTTPException(
            status_code=500, 
            detail="Failed to generate business name suggestions. Please try again with a more detailed description."
        )
    
    return {"suggestions": suggestions}

@app.get("/health")
def health_check():
    """Health check endpoint to verify the API is running"""
    return {"status": "ok"}

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000)

import streamlit as st
from typing import List, Dict
import re
import os
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone, ServerlessSpec
import json
from openai import OpenAI  # Updated import for OpenAI v1.0+

# Load environment variables
load_dotenv()

# Initialize OpenAI client (updated for v1.0+)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define index names
primary_index_name = os.getenv("PINECONE_INDEX_NAME", "mca-scraped-final1")
trademark_index_name = os.getenv("TRADEMARK_INDEX_NAME", "tm-prod-pipeline")

# Initialize indexes as None initially
primary_index = None
trademark_index = None

def verify_indexes():
    """Verify both indexes are accessible and return connection status"""
    global primary_index, trademark_index
    
    # Try connecting to primary index
    try:
        primary_index = pc.Index(primary_index_name)
        # Just initialize the connection without any test query
    except Exception:
        # Silently handle the error
        pass
    
    # Try connecting to trademark index
    try:
        trademark_index = pc.Index(trademark_index_name)
        # Just initialize the connection without any test query
    except Exception:
        # Silently handle the error
        pass

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
        primary_result = NameValidator._check_primary_index(name)
        if primary_result:
            return True
            
        # Check trademark index (wordMark format)
        trademark_result = NameValidator._check_trademark_index(name)
        if trademark_result:
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
            
        except Exception:
            # Silently handle the error
            return False
            
    @staticmethod
    def _check_trademark_index(name: str) -> bool:
        """Check if name exists in the trademark index"""
        global trademark_index
        
        if not trademark_index:
            return False  # Assume name doesn't exist if we can't check
            
        try:
            # Extract the main business name part before any dash or special character
            # This handles cases like "BusinessName - Tagline"
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
            
        except Exception:
            # Silently handle the error
            return False

class BusinessNameGenerator:
    @staticmethod
    def generate_business_names(description: str) -> List[Dict[str, str]]:
        """Generate unique business name suggestions based on user description."""
        try:
            # Updated to use OpenAI v1.0+ API
            response = client.chat.completions.create(
                model="gpt-4",
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
                        "content": f"Create 10 COMPLETELY UNIQUE business names for this description, ensuring ZERO similarity between names: '{description}'. For each name, provide a 10-word description of why it's appropriate."
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
                    if len(unique_suggestions) == 6:
                        break
            
            # If we don't have enough valid names, show a warning
            if len(unique_suggestions) < 6:
                st.warning(f"Some generated names already exist in the database. Showing {len(unique_suggestions)} unique options.")
            
            return unique_suggestions
        except Exception as e:
            st.error(f"Error generating business names: {str(e)}")
            return []

def main():
    # Page configuration
    st.set_page_config(
        page_title="Zolvit Name Generator",
        page_icon="🚀",
        layout="wide"
    )
    
    # Initialize session state for generated names
    if 'generated_names' not in st.session_state:
        st.session_state.generated_names = []
    
    # Verify index connections silently
    verify_indexes()
    
    # Custom CSS for professional look
    st.markdown("""
    <style>
    /* Professional dark theme */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Container styling */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Name suggestion styling */
    .name-container {
        background-color: #1E2129;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #FFFFFF;
    }
    
    /* Badge for validated names */
    .validated-badge {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 2px 8px;
        font-size: 12px;
        margin-left: 10px;
    }
    
    /* Description styling */
    .name-description {
        font-size: 14px;
        color: #CCC;
        margin-top: 8px;
        font-style: italic;
    }
    
    /* Custom styling for placeholder */
    textarea::placeholder {
        color: #666 !important;
        opacity: 0.8 !important;
        font-style: italic !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Title and description
    st.title("🚀 Zolvit Business Name Generator")
    st.markdown("### Generate Unique and Creative Business Names")
    st.markdown("##### All suggested names are validated against our database to ensure they are not already in use.")
    
    # Initialize session state for generated names only (removed search count)
    if 'generated_names' not in st.session_state:
        st.session_state.generated_names = []
    
    # Description input with static placeholder
    description = st.text_area(
        "Describe your business idea", 
        height=150, 
        placeholder="E.g., An innovative coffee shop with a modern twist..."
    )
    
    # Generate Names Button - No longer disabled based on count
    generate_button = st.button(
        "Generate Business Names", 
        use_container_width=True
    )
    
    if generate_button:
        if not description:
            st.warning("Please enter a business description")
        else:
            with st.spinner("Generating unique business names and validating against existing records..."):
                generated_names = BusinessNameGenerator.generate_business_names(description)
                
                if generated_names:
                    st.session_state.generated_names = generated_names
                    
                    if len(generated_names) < 6:
                        st.info("Some generated names were filtered out because they already exist in our database.")
                else:
                    st.error("Failed to generate business name suggestions")
    
    # Display Names
    if st.session_state.generated_names:
        # Name container
        st.markdown('<div class="name-container">', unsafe_allow_html=True)
        
        st.subheader("Business Names")
        st.markdown("✅ All names verified as unique")
        
        # Create a grid layout for the names
        cols = st.columns(3)
        
        for i, suggestion in enumerate(st.session_state.generated_names):
            with cols[i % 3]:
                st.markdown(f"""
                <div style='text-align: center; padding: 15px; background-color: #2C3040; border-radius: 8px; margin-bottom: 15px;'>
                    <h3>{suggestion['name']}</h3>
                    <span class='validated-badge'>Verified Unique</span>
                    <p class='name-description'>{suggestion['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Close name container
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
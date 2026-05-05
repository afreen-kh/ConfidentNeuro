from dotenv import load_dotenv
load_dotenv()
import os
import time
import json
import csv
import pandas as pd
from Bio import Entrez
import google.generativeai as genai
from typing import List, Dict

# Configuration
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMAIL = "afreen@example.com"  # Required by Entrez

# Setup
Entrez.email = EMAIL
Entrez.api_key = NCBI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)
# Using gemini-2.5-flash as gemini-2.0-flash is currently restricted for this user
model = genai.GenerativeModel('models/gemini-2.5-flash')

CATEGORIES = {
    "Category 1: Diagnosis": {"mesh": "Alzheimer Disease/diagnosis", "prefix": "DIAG"},
    "Category 2: Disease Staging": {"mesh": "Alzheimer Disease/pathology", "prefix": "STAG"},
    "Category 3: Treatment & Management": {"mesh": "Alzheimer Disease/therapy", "prefix": "TRET"},
    "Category 4: Biomarkers": {"mesh": "Alzheimer Disease/cerebrospinal fluid OR Alzheimer Disease/blood", "prefix": "BIOM"},
    "Category 5: Prognosis & Risk Factors": {"mesh": "Alzheimer Disease/epidemiology", "prefix": "PROG"}
}

def fetch_pubmed_abstracts(category_name: str, mesh_query: str, count: int = 10) -> List[Dict]:
    """Retrieves abstracts from PubMed based on MeSH terms."""
    print(f"\nSearching PubMed for '{category_name}'...")
    
    search_query = f"({mesh_query}) AND (\"2020\"[Date - Publication] : \"2025\"[Date - Publication])"
    
    try:
        handle = Entrez.esearch(db="pubmed", term=search_query, retmax=count)
        record = Entrez.read(handle)
        handle.close()
        
        pmids = record.get("IdList", [])
        if not pmids:
            print(f"No results found for {category_name}.")
            return []
            
        print(f"Found {len(pmids)} PMIDs. Fetching details...")
        
        handle = Entrez.efetch(db="pubmed", id=pmids, rettype="xml", retmode="text")
        records = Entrez.read(handle)
        handle.close()
        
        abstracts = []
        for article in records.get("PubmedArticle", []):
            pubmed_data = article.get("MedlineCitation", {})
            pmid = str(pubmed_data.get("PMID", ""))
            article_data = pubmed_data.get("Article", {})
            title = article_data.get("ArticleTitle", "")
            
            abstract_text = ""
            abstract = article_data.get("Abstract", {})
            if "AbstractText" in abstract:
                text_list = abstract["AbstractText"]
                if isinstance(text_list, list):
                    abstract_text = " ".join([str(t) for t in text_list])
                else:
                    abstract_text = str(text_list)
            
            if abstract_text:
                abstracts.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract_text,
                    "mesh_terms": mesh_query,
                    "category": category_name
                })
        
        return abstracts[:count]
    except Exception as e:
        print(f"Error fetching PubMed data: {e}")
        return []

def generate_questions_with_gemini(abstract_data: Dict) -> List[Dict]:
    """Uses Gemini to generate 3 clinical questions from an abstract."""
    print(f"Generating questions for PMID: {abstract_data['pmid']}...")
    
    prompt = f"""
    You are a senior clinical neurologist. Generate 3 specific clinical decision support questions from this abstract.
    
    Title: {abstract_data['title']}
    Abstract: {abstract_data['abstract']}
    Category: {abstract_data['category']}
    
    Output JSON array:
    [
        {{"question": "...", "ground_truth_answer": "...", "difficulty": "easy/medium/hard"}},
        ...
    ]
    """
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Shorten abstract to around 1000 chars to avoid TPM issues
            abstract_snippet = abstract_data['abstract'][:1000]
            
            prompt = f"Ref: {abstract_data['title']}\nAbstract: {abstract_snippet}\nCategory: {abstract_data['category']}\nTask: Generate 3 clinical decision support questions with ground truth and difficulty (easy/medium/hard). Output MUST be JSON array: [{{'question': '...', 'ground_truth_answer': '...', 'difficulty': '...'}}]"

            response = model.generate_content(prompt)
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
                
            questions = json.loads(text)
            
            final_questions = []
            prefix = abstract_data.get("prefix", "QUES")
            for i, q in enumerate(questions):
                q_id = f"{prefix}_{abstract_data['pmid'][-3:]}_{i+1:03d}"
                final_questions.append({
                    "question_id": q_id,
                    "category": abstract_data['category'],
                    "question": q.get("question", ""),
                    "ground_truth_answer": q.get("ground_truth_answer", ""),
                    "source_pmid": abstract_data['pmid'],
                    "mesh_terms": abstract_data['mesh_terms'],
                    "difficulty": q.get("difficulty", "medium").lower()
                })
            
            # 60 second delay between successful calls to be very safe
            time.sleep(60) 
            return final_questions
            
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 65 * (attempt + 1)
                print(f"Rate limit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"Error on PMID {abstract_data['pmid']}: {e}")
                time.sleep(15)
    print(f"Skipping PMID {abstract_data['pmid']} after max retries.")
    return []

def main():
    all_questions = []
    
    # Check if we have partial results to resume (simple version)
    if os.path.exists("benchmark_150.json"):
        with open("benchmark_150.json", "r") as f:
            try:
                all_questions = json.load(f)
                print(f"Resuming with {len(all_questions)} existing questions.")
            except:
                pass

    # Fixed goal: 30 questions per category (10 abstracts with 3 questions each)
    TARGET_PER_CAT = 30

    for cat_name, data in CATEGORIES.items():
        # Count current questions for this category
        cat_count = sum(1 for q in all_questions if q["category"] == cat_name)
        if cat_count >= TARGET_PER_CAT:
            print(f"'{cat_name}' already complete with {cat_count} questions.")
            continue
            
        print(f"'{cat_name}' needs {TARGET_PER_CAT - cat_count} more questions.")
        
        # Fetch more PMIDs to find new abstracts
        # We start by fetching more than 10 to ensure we have enough buffer
        abstracts = fetch_pubmed_abstracts(cat_name, data["mesh"], count=20)
        
        for abstract in abstracts:
            if cat_count >= TARGET_PER_CAT:
                break
                
            # Check if PMID already processed
            if any(q["source_pmid"] == abstract["pmid"] for q in all_questions):
                continue
                
            abstract["prefix"] = data["prefix"]
            questions = generate_questions_with_gemini(abstract)
            if questions:
                # Limit to 3 questions per abstract according to previous logic
                q_to_add = questions[:3]
                all_questions.extend(q_to_add)
                cat_count += len(q_to_add)
                
                # Save after each successful abstract
                with open("benchmark_150.json", "w") as f:
                    json.dump(all_questions, f, indent=4)
                pd.DataFrame(all_questions).to_csv("benchmark_150.csv", index=False)
                print(f"'{cat_name}' progress: {cat_count}/{TARGET_PER_CAT}")

    print("\nBenchmark generation complete!")
    print(f"Final Count: {len(all_questions)}")

if __name__ == "__main__":
    main()

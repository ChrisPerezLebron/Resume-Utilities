from haystack.components.embedders import SentenceTransformersTextEmbedder
import numpy as np 
import os 
import fitz
import pickle

# Setup text embedder
text_embedder = SentenceTransformersTextEmbedder(model='sentence-transformers/all-MiniLM-l6-v2')
text_embedder.warm_up() 

# Embed the list of skills
    #skills for Resume 3.0
result = text_embedder.run("""
                Java
                Agile Development (mentioned 5x)
                Web UI Development (HTML, XML, XSLT, JSON)
                Neo4j 
                Schema 
                Python 
                C/C++ 
                Eclipse 
                JUnit
                test driven development 
                refactoring 
                modular deisgn/maintainable 
                Collaboration
            """)

# Extract actual embedding as a Numpy array 
skills_embedding = np.array(result['embedding'])

# Convert the embedding to unit length to speed up cosine similarity computation later
unit_skills_embedding = skills_embedding/np.linalg.norm(skills_embedding)

# Helper function fo resume text extraction
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Create list of resumes 
resume_names = [] 
resumes = []
folder_path = "/Users/christopherperezlebron/Documents/Resumes"
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        text = extract_text_from_pdf(file_path)
        resume_names.append(filename)
        resumes.append(text)

# Make np matrix for unit length resume embeddings (NumResumes x Embedding Dimension)
resume_embeddings = np.zeros(shape=(len(resumes), len(result['embedding'])))

# Create embeddings for resumes
for i in range(len(resumes)):
    cache_path = "cache/"
    file_path = cache_path+resume_names[i].replace(".pdf", ".pkl")

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            resume_embedding = pickle.load(f)
        print(f"Loaded cached {resume_names[i]} from {cache_path}.")
    else:
        print(f"Building {resume_names[i]} (this may take a while)…")
        result = text_embedder.run(resumes[i])
        resume_embedding = np.array(result['embedding'])
        resume_embedding = resume_embedding/np.linalg.norm(resume_embedding)
        with open(file_path, "wb") as f:
            pickle.dump(resume_embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {resume_names[i]} cache to {cache_path}.")
        
    resume_embeddings[i] = resume_embedding
      
# Obtain Cosine Similarity matrix by taking dot product 
similarities = np.dot(resume_embeddings, unit_skills_embedding)

best_resumes = np.argsort(similarities, axis=0)

# print best resumes w/ cosine similarity score
for i in range(len(best_resumes)-1, -1, -1):
    print(resume_names[best_resumes[i]], similarities[best_resumes[i]])
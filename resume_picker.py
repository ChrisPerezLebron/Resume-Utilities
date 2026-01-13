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
Python
Object-oriented programming
REST APIs
Spring Boot
Angular
JavaScript
SQL databases
Backend service development
API development
Frontend component development
Scalable UI development
Performance optimization
Agile methodologies
Peer code reviews
Debugging
Software testing
CI/CD
CI/CD pipeline maintenance
Technical documentation
Collaboration
Problem solving
Communication skills
AI/ML concepts
TensorFlow
scikit-learn
AWS
Cloud platforms
Git
Docker
Internship experience
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



# Create or load embeddings 
resume_names = [] 
resumes = []
# resume_folder_path = "/Users/christopherperezlebron/Documents/Resumes"
resume_folder_path = "./resumes"
for filename in os.listdir(resume_folder_path):
    cache_path = "cache/"
    pkl_file_path = cache_path+filename.replace(".pdf", ".pkl")

    #if this resume's unit embedding has been cached, load it
    if filename.lower().endswith(".pdf") and os.path.exists(pkl_file_path):
        with open(pkl_file_path, "rb") as f:
            resume_embedding = pickle.load(f)
        print(f"Loaded cached {filename} from {pkl_file_path}.")

        #Store the embedding and the corresponding file name
        resume_names.append(filename)
        resumes.append(resume_embedding)
    #otherwise, make it 
    elif filename.lower().endswith(".pdf"):
        resume_file_path = os.path.join(resume_folder_path, filename)
        text = extract_text_from_pdf(resume_file_path)
        result = text_embedder.run(text)
        resume_embedding = np.array(result['embedding'])
        resume_embedding = resume_embedding/np.linalg.norm(resume_embedding)
        with open(pkl_file_path, "wb") as f:
            pickle.dump(resume_embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {filename} cache to {pkl_file_path}.")

        # Store the embedding and the corresponding file name
        resume_names.append(filename)
        resumes.append(resume_embedding)

# Make np matrix for unit length resume embeddings (NumResumes x Embedding Dimension)
resume_embeddings = np.array(resumes, dtype=float)
      
# Obtain Cosine Similarity matrix by taking dot product 
similarities = np.dot(resume_embeddings, unit_skills_embedding)

best_resumes = np.argsort(similarities, axis=0)

# print best resumes w/ cosine similarity score
for i in range(len(best_resumes)-1, -1, -1):
    print(resume_names[best_resumes[i]], similarities[best_resumes[i]])
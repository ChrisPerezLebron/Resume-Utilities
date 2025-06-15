from haystack.components.embedders import SentenceTransformersTextEmbedder
import numpy as np 
import os 
import fitz

#Goal:
    #based on the sentence embedding of a list of skills, argsort the following resume items 
        #related course experience
        #work experience bullet points 
        #projects & corresponding bullet points


# Setup text embedder
text_embedder = SentenceTransformersTextEmbedder(model='sentence-transformers/all-MiniLM-l6-v2')
text_embedder.warm_up() 

# Embedd the list of skills
skills_result = text_embedder.run("""
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
skills_embedding = np.array(skills_result['embedding'])

# Normalize the skills embedding (to allow for more efficient cosine similarity comparisons)
skills_embedding_unit = skills_embedding/np.linalg.norm(skills_embedding)

courses = [
    "CMSC351 - Algorithms",
    "STAT400 - Applied Probability and Statistics I",
    "MATH181 - Calculus I",
    "MATH182 - Calculus II",
    "CMSC203 - Computer Science I - Object Oriented Programming (Java)",
    "CMSC204 - Computer Science II - Data Structures and Algorithms (Java)",
    "CMSC426 - Computer Vision (Python)",
    "PHYS203 - General Physics Non Engineering I",
    "ENGL102HC - Honors Critical Reading, Writing, and Research", 
    "CMSC421 - Intro to Artificial Intelligence (Python)",
    "CMSC430- Intro to Compilers (Racket, x86)",
    "CMSC216 - Intro to Computer Systems (C/C++, ARM Assembly)",
    "CMSC320 - Intro to Datascience (Python)",
    "ENES100 - Intro to Engineering Design",
    "CMSC422 - Intro to Machine Learning (Python)", 
    "MATH206 - Intro to Matlab",
    "CMSC470 - Intro to Natural Language Processing (Python)",
    "CMSC416 - Intro to Parallel Computing (C/C++, Cuda)",
    "CMSC140 - Intro to Programming (C++)",
    "CMSC246 - Intro to SQL Using Oracle",
    "MATH240 - Linear Algebra (Matlab)",
    "CMSC330 - Organization of Programming Languages (Python, OCamel, Rust)",
    "ENGL393 - Technical Writting",
]
# Embedd all courses as unit length word embeddings into a single matrix
courses_matrix_unit = np.zeros(shape=(len(courses), len(skills_embedding)))
for i, course in enumerate(courses):
    # Get embedding for this course  
    course_embedding = np.array(text_embedder.run(course)['embedding'])

    # Normalize the embedding for cosine similarity later
    courses_matrix_unit[i] = course_embedding/np.linalg.norm(course_embedding)


# Perform cosine similarity
course_similarity = np.dot(courses_matrix_unit, skills_embedding_unit)

# argsort the courses by cosine similarity to provided skills 
best_courses = np.argsort(course_similarity, axis=0)

# Formating for output 
print("")
print("================================================Courses START================================================")

# print best resumes w/ cosine similarity score
for i in range(len(best_courses)-1, -1, -1):
    print(course_similarity[best_courses[i]], courses[best_courses[i]][:112])
# Formatting for output
print("================================================Courses END================================================")
print("")


# List of work experience bullet points
work_experience = [
    "Drove full-stack development of a robust HR analytics platform across the full Software Development Life Cycle (SDLC), from requirements gathering and architecture design to testing and deployment, delivering production-ready features in Agile sprints and enhancing HR insights accessibility by 40%.",
    "Built a consumer-facing HR analytics dashboard using JavaScript and Chart.js, translating Neo4j knowledge graph CRUD operations into real-time time-series visualizations. Reduced HR decision latency by 40% and improved anomaly detection by 65% through continuous change monitoring.",
    "Conducted product research and requirements analysis on Obsidian, using findings to inform the system design and Neo4j schema architecture for an HR knowledge graph application. This accelerated delivery of a streamlined internal talent-tracking web app by a 3-person full stack team, reducing development time by 25%.",
    "Engineered retrieval-augmented generation (RAG) pipelines for our HR knowledge graph application, implementing bidirectional relationship extraction and token optimization strategies that improved context relevance by 50% while troubleshooting GPU acceleration to reduce model inference latency by 30%.",
    "Programmed Arduino microcontrollers in C/C++ to design and debug digital circuits (button-controlled LED matrices, sensor-triggered fans), implementing GPIO manipulation and interrupt handling to advance foundational hardware and embedded systems knowledge.",
    "Led Docker containerization and deployment of a consumer-facing HR analytics platform to AWS, collaborating cross-functionally with engineers, cybersecurity, and C-suite stakeholders to improve application availability and enable scalable production rollout.", #= to next 
    "Deployed a containerized application to AWS using Docker on a Linux instance, resolving port conflicts over TCP/IP and accelerating future deployments by 40%. Simulated the production environment locally with VMware and VirtualBox, and managed AWS servers via CLI tools.", #= to prior
    "Delivered maintainable Neo4j CRUD operations to manage 100+ employee records, enhancing backend data handling and platform scalability.", 
    "Developed knowledge graph embeddings using Neo4j's vector indexing capabilities, leveraging cosine similarity search that improved node relationship discovery accuracy by 55% for our AI-powered HR analytics web application.",
    "Automated JSON data ingestion handling 100+ daily HTTP requests, integrating schema validation to reduce parsing errors by 65%. Connected the pipeline to a RESTful API built with Python and Flask for real-time updates to the Neo4j database.",
    "Designed and built responsive UI components with Bootstrap and jQuery, integrating asynchronous HTTP handlers to efficiently render backend JSON payloads and reduce load times by 35% in large-scale knowledge graph visualizations.",
    "Developed and configured Neo4j vector search indexes to support graph embeddings, enabling semantic similarity comparisons that enhanced response accuracy in the question-answering RAG (Retrieval-Augmented Generation) ML pipeline by 75%.",
    "Engineered HR knowledge graph solutions using Neo4j and Cypher queries, collaborating with three software engineers to implement graph-based search and Natural Language Processing capabilities that reduced data retrieval time by 65%",
    "Developed Python/Flask REST APIs with OpenAI integration for resume parsing, implementing prompt engineering with Jinja templates to extract 15+ entity types from documents, improving data extraction accuracy by 40% while ingesting data far faster than manual input.",
    "Engineered a full-stack knowledge graph application with a Python/Flask backend and JavaScript/Bootstrap frontend, implementing dynamic AJAX/jQuery UI components and real-time form validation that cut CRUD operation time by 30%.", #= to next two
    "Engineered a full-stack web application using a Python/Flask backend and a JavaScript/Bootstrap frontend, collaborating with three Software Engineers to implement 50+ REST API endpoints handling JSON payloads and HTTP requests, reducing CRUD operation time by 30% through dynamic AJAX components and modular design patterns", #= to prior and next
    "Engineered a full-stack knowledge graph application using Python/Flask backend and JavaScript/Bootstrap frontend, collaborating with 3 software engineers to implement 15+ REST API endpoints for CRUD operations on Neo4j graph databases, reducing data retrieval latency by 40% through optimized Cypher queries.", #= to prior two
    "Designed a CSV data ingestion pipeline using Python, Flask, and Pandas, creating an adjacency list parser that reduced data loading complexity by 70% while implementing null-handling safeguards that significantly increased data integrity.",
    "Revamped user interfaces with Bootstrap/CSS, implementing modal-based editing and autocomplete search that decreased user errors by 45% in node-relationship management workflows.", 
    "Integrated OpenAI API for question answering, retrieval-augmented generation, leveraged Jinja to perform prompt engineering, which improved response accuracy by 80% while implementing usage caps to control API costs.",
    "Created comprehensive repository wiki documentation for graph schemas and RESTful API endpoints, improving onboarding efficiency by 50% for new developers while standardizing data ingestion protocols.",
    "Developed an elegant graph view using HTML5, CSS, JavaScript, Bootstrap, and Alchemy.js while using AJAX and RESTful API calls for dynamic updates, enabling real-time graph updates that accelerated HR data visualization by 90%.",
    "Implemented schema validation with Python, Flask, and Neo4j, creating node-type checking algorithms that reduced data inconsistencies by 80% in knowledge graph updates.",
    "Refactored 5,000+ lines of Python/JavaScript code in an Agile environment using Git, resolving 15+ critical bugs while improving maintainability and reliability by 90%.",
    "Researched Hugging Face models and vector embeddings to utilize in retrieval augmented generation against the data in our knowledge graph application, prototyping local Large Language Model solutions that demonstrated 50% faster query resolution for candidate search scenarios.",
    "Designed responsive user interfaces with Bootstrap/Jinja templates, implementing dynamic layouts by leveraging AJAX and jQuery to modify page layouts without necessitating page refresh, resulting in a more satisfying user experience.",
    "Visualized Human Resource information highlighting skills, certifications, and clearances held by the internal workforce using HTML5, CSS, JavaScript, and Alchemy.js, producing relationship visualization components that decreased user errors by 25% in node creation workflows.",
    "Developed knowledge graph-based search and Natural Language Processing features including autocomplete functionality using Cypher query language, enabling complex relationship mapping across 200+ nodes which improved HR data query efficiency by 60%",
    "Integrated Flask REST APIs with JavaScript frontend using AJAX/jQuery, developing dynamic node/relationship management interfaces that reduced CRUD operation time by 30% through real-time form validation and modal-based editing.",
    "Containerized applications using Docker for AWS cloud deployment while documenting 20+ API endpoints and code functionality, improving onboarding efficiency by 50% for new Software Engineering hires.",
    "Created data ingestion systems using Pandas to process CSV data into Neo4j knowledge graphs, streamlining the ingestion of over 200+ nodes and 100+ relationships.",
    "Led frontend bug resolution and feature enhancements in Agile environment using GitHub, refactoring 5K+ lines of front end JavaScript, HTML5, and CSS code into modular components that improved maintainability by 50%.", 
]

# Embedd all bullet points as unit length word embeddings into a single matrix
work_experience_matrix_unit = np.zeros(shape=(len(work_experience), len(skills_embedding)))
for i, bullet in enumerate(work_experience):
    # Get embedding for this bullet  
    bullet_embeddding = np.array(text_embedder.run(bullet)['embedding'])

    # Normalize the embedding for cosine similarity later
    work_experience_matrix_unit[i] = bullet_embeddding/np.linalg.norm(bullet_embeddding)


# Perform cosine similarity
work_experience_similarity = np.dot(work_experience_matrix_unit, skills_embedding_unit)

# Argsort the bullet points 
best_work_experience = np.argsort(work_experience_similarity, axis=0)

# Formating for output 
print("")
print("================================================Work Experience START================================================")

# print best resumes w/ cosine similarity score
for i in range(len(best_work_experience)-1, -1, -1):
    print(work_experience_similarity[best_work_experience[i]], work_experience[best_work_experience[i]][:112])
# Formatting for output
print("================================================Work Experience END================================================")
print("")
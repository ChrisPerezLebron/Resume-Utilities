from haystack.components.embedders import SentenceTransformersTextEmbedder
import numpy as np 

text_embedder = SentenceTransformersTextEmbedder(model='sentence-transformers/all-MiniLM-l6-v2')
text_embedder.warm_up() 

skills = text_embedder.run("""
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
skills = np.array(skills['embedding'])

resume1 = text_embedder.run("""Christopher Perez Lebron
240-802-9459 | ChrisPerezLebron@gmail.com | linkedin.com/in/ChrisPerezLebron | github.com/ChrisPerezLebron
Education
University of Maryland Bachelor of Science in Computer Science, Machine Learning Track, GPA: 3.90 Relevant Coursework Compilers, Data Structures, Parallel Programming, Object Oriented Programming
College Park, MD
Aug. 2023 – May 2025
Montgomery College Rockville, MD
Associates of Arts in Computer Science, GPA: 3.93 Aug. 2020 – May 2023
Experience
Software Engineer Intern May 2023 – Aug. 2023 | Jan. 2024 | May 2024 – Aug. 2024 | Jan. 2025
Acclaim Technical Services Hanover, MD
• Refactored 5,000+ lines of Python/JavaScript code in an Agile environment using Git, resolving 15+ critical bugs
while improving maintainability by 90%.
• Engineered a full-stack knowledge graph application with a Python/Flask backend and JavaScript/Bootstrap
frontend, implementing dynamic AJAX/jQuery UI components and real-time form validation that cut CRUD
operation time by 30%.
• Developed an elegant graph view using HTML, CSS, JavaScript, Bootstrap, and Alchemy.js while using AJAX and
RESTful API calls for dynamic updates, enabling real-time graph updates that accelerated HR data visualization
by 90%.
• Automated JSON data ingestion processing 100+ daily HTTP requests, implementing schema validation that
reduced parsing errors by 65% while integrating with our RESTful API built using Python/Flask for real-time
graph database updates.
• Developed and configured Neo4j vector search indexes to support graph embeddings, enabling semantic similarity
comparisons that enhanced response accuracy in the question-answering RAG (Retrieval-Augmented Generation)
ML pipeline by 75%.
• Programmed Arduino microcontrollers in C/C++ to design and debug digital circuits (button-controlled LED
matrices, sensor-triggered fans), implementing GPIO manipulation and interrupt handling to advance foundational
hardware and embedded systems knowledge.
• Led Docker containerization of a consumer-facing HR analytics platform through cross-functional collaboration
with engineers, cybersecurity, and C-suite stakeholders, establishing code review protocols that boosted system
stability by 40% and delivering maintainable Neo4j CRUD operations for 100+ employee records.
• Created comprehensive repository wiki documentation for graph schemas and RESTful API endpoints, improving
onboarding efficiency by 50% for new developers while standardizing data ingestion protocols.
Projects
Graphical Town Explorer | Java, JUnit, JavaFX
• Implemented Test-Driven Development (TDD) and quality assurance protocols using JUnit 5 Jupiter for a
Java-based graph routing application, designing 36 test cases covering Dijkstra’s algorithm, edge validation, and
exception handling that achieved 100% critical path coverage and reduced graph operation defects by 75%.
• Architected maintainable Java object-oriented programming (OOP) solution for graphical town mapping software
by implementing a custom graph data structure (Town/Road/Graph classes) with encapsulation and inheritance,
solving complex pathfinding challenges through Dijkstra’s algorithm that processed 100+ node networks, 40%
faster runtime than the adjacency matrix approach.
• Engineered modular and maintainable Java solution for town routing system featuring error handling (25+ edge
cases), SOLID-compliant class hierarchy, and robust graph CRUD operations, that scaled to 100+ nodes while
maintaining 100% specification compliance.
Technical Skills
Languages: Python, Java, JavaScript, C/C++, SQL, Racket, x86 Assembly, ARM Assembly, HTML/CSS, Rust
Frameworks: React, Flask, JUnit, OpenMP, MPI, CUDA
Developer Tools: Git, Docker, VS Code, Eclipse, gdb, lldb, Make, CMake, SLURM
Libraries: Pandas, NumPy, Matplotlib, PyTorch, scikit-learn, OpenCV, jQuery, Bootstrap""")
resume1 = np.array(resume1['embedding'])

resume2 = text_embedder.run("""Christopher Perez Lebron
240-802-9459 | ChrisPerezLebron@gmail.com | linkedin.com/in/ChrisPerezLebron | github.com/ChrisPerezLebron
Education
University of Maryland Bachelor of Science in Computer Science, Machine Learning Track, GPA: 3.90 Relevant Coursework Compilers, Data Structures, Parallel Programming, Object Oriented Programming
College Park, MD
Aug. 2023 – May 2025
Montgomery College Rockville, MD
Associates of Arts in Computer Science, GPA: 3.93 Aug. 2020 – May 2023
Experience
Software Engineer Intern May 2023 – Aug. 2023 | Jan. 2024 | May 2024 – Aug. 2024 | Jan. 2025
Acclaim Technical Services Hanover, MD
• Developed and configured Neo4j vector search indexes to support graph embeddings, enabling semantic similarity
comparisons that enhanced response accuracy in the question-answering RAG (Retrieval-Augmented Generation)
ML pipeline by 75%.
• Deployed a containerized application to AWS over TCP/IP using Docker on a Linux instance, resolving port
conflicts and accelerating future deployment cycles by 40%. Simulated the environment locally using VMware and
VirtualBox, and managed the AWS server via command-line tools.
• Programmed Arduino microcontrollers in C/C++ to design and debug digital circuits (button-controlled LED
matrices, sensor-triggered fans), implementing GPIO manipulation and interrupt handling to advance foundational
hardware and embedded systems knowledge.
• Created comprehensive repository wiki documentation for graph schemas and RESTful API endpoints, improving
onboarding efficiency by 50% for new developers while standardizing data ingestion protocols.
• Refactored 5,000+ lines of Python/JavaScript code in an Agile environment using Git, resolving 15+ critical bugs
while improving maintainability by 90%.
• Automated JSON data ingestion processing 100+ daily HTTP requests, implementing schema validation that
reduced parsing errors by 65% while integrating with our RESTful API built using Python/Flask for real-time
graph database updates.
• Developed an elegant graph view using HTML, CSS, JavaScript, Bootstrap, and Alchemy.js while using AJAX and
RESTful API calls for dynamic updates, enabling real-time graph updates that accelerated HR data visualization
by 90%.
• Engineered a full-stack web application using a Python/Flask backend and a JavaScript/Bootstrap frontend,
collaborating with three Software Engineers to implement 50+ REST API endpoints handling JSON payloads and
HTTP requests, reducing CRUD operation time by 30% through dynamic AJAX components and modular design
patterns.
Projects
Racket Compiler | Racket, x86, C
• Developed a Racket compiler leveraging x86 Assembly and C runtime integration, implementing stack-based calling
conventions with tail-call optimization that reduced recursive function overhead by 40% in benchmark tests.
• Debugged memory pointer operations using gdb/lldb for x86 instruction tracing, eliminated 15+ segmentation
faults in memory pointer operations through systematic watchpoint analysis.
• Utilized Test-Driven Development (TDD) with the RackUnit framework to guide compiler development, applying
functional programming principles. Wrote 150+ test cases covering 15+ Racket language features (e.g., arithmetic,
I/O, pattern matching), which ensured correctness and reduced logic errors by 95% through thorough edge-case
coverage.
Technical Skills
Languages: Python, Java, JavaScript, C/C++, SQL, Racket, x86 Assembly, ARM Assembly, HTML/CSS, Rust
Frameworks: React, Flask, JUnit, OpenMP, MPI, CUDA
Developer Tools: Git, Docker, VS Code, gdb, lldb, Make, CMake, SLURM
Libraries: Pandas, NumPy, Matplotlib, PyTorch, scikit-learn, OpenCV, jQuery, Bootstrap""") 
resume2 = np.array(resume2['embedding'])

cos_sim_resume1 = np.dot(skills, resume1)/(np.linalg.norm(skills)*np.linalg.norm(resume1))
cos_sim_resume2 = np.dot(skills, resume2)/(np.linalg.norm(skills)*np.linalg.norm(resume2))


print(f'cos_sim_resume1 --> {cos_sim_resume1}')
print(f'cos_sim_resume2 --> {cos_sim_resume2}')

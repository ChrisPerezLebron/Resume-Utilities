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


# For projects we use a list of tuple (projectTitle, [bullet points])
    #but we embedd projectTitle+bulletPoint for each bullet point to find most relevant project experience
projects = [
    ("SQL Data Analysis | SQL, Pyhon", [
        "Python",
        "SQLite",
        "Pandas",
        "SQL",
        "SQL queries",
        "Data Analysis",
        "Data Aggregation",
        "SQL Table Creation",
        "SQL Table Joining (using inner join)",
        "Pandas DataFrame",
        "Data Filtering",
        "Data Sorting",
        "Data Summarization",
        "Relational Databases",
        "Data Manipulation",
        "SQL Functions",
        "Data Exploration",
        "SQL Table Design",
        "Large Kaggle Datasets",
        "Data Cleaning"
    ]),
    ("Wine Data Analysis | Python, Numpy, Matplotlib", [
        "python",
        "numpy",
        "pandas",
        "sklearn",
        "k-fold cross-validation",
        "matplotlib", 
        "scikit-learn",
        "cross_val_score",
        "StratifiedKFold",
        "StandardScaler", 
        "KNeighborsClassifier",
        "DecisionTreeClassifier", 
        "LogisticRegression", 
        "RandomForestClassifier",
        "data preprocessing", 
        "data standardization",
        "machine learning", 
        "classification", 
        "model evaluation", 
        "data splitting", 
        "feature engineering", 
        "feature importance", 
        "data visualization", 
        "correlation matrix", 
        "data analysis", 
        "data cleaning",
        "Exploratory Data Analysis (EDA)", 
        "CSV file handling",
        "binary classification", 
        "model training", 
        "model testing", 
        "data science", 
        "supervised learning",
        "model selection"
    ]),
    ("Regression Using Neural Networks | Python, Numpy, Sklearn", [
        "linear regression with polynomial features", 
        "gradient descent from scratch to train linear regression model",   
        "modified regression model to perform binary classification",
        "researched kernel functions to enable the seperation of non linear data using  linear decision boundaries",
        "trained neural network using pytorch tensors on california housing data",
        "machine learning",
        "Python", 
        "NumPy", 
        "Matplotlib",
        "scikit-learn",
        "PyTorch",
        "Linear Regression", 
        "Polynomial Regression", 
        "Ridge Regression", 
        "Lasso Regression", 
        "ElasticNet Regression", 
        "Mean Squared Error", 
        "R-squared", 
        "Data Visualization", 
        "Machine Learning",
        "Model Evaluation",  
        "PolynomialFeatures", 
        "StandardScaler", 
        "Adam Optimizer", 
        "Loss Function", 
        "Classification", 
        "Regularization", 
        "Representation Learning", 
        "Overfitting", 
        "Underfitting", 
        "Regression Modeling", 
        "Custom ML Model Implementation", 
        "Hyperparameter Tuning" 
    ]),
    ("Walmart Sales Forecasting | Python, Pandas, NumPy, Scikit-Learn", [
        "Led a 3-member data science team to analyze 3 years of Walmart sales data (6,435 records across 45 stores) using Python, pandas, NumPy, and scikit-learn; applied statistical hypothesis testing to uncover key sales drivers.",
        "Applied machine learning models (Decision Tree, Linear Regression, Random Forest) using K-fold cross-validation to analyze weekly sales data; identified store ID and CPI as key predictors, achieving high R-squared  and low forecast error.",
        "Designed and interpreted 15+ advanced data visualizations using Matplotlib and pandas, including correlation matrices, multi-variable histograms, and loss curves, to analyze Walmart’s time-series sales data and guide feature engineering for machine learning models.",
        "Collaborated with other talented software engineers (team of 3)",
        "NumPy",
        "SciPy",
        "scikit-learn",
        "Data Cleaning",
        "Exploratory Data Analysis",
        "Linear Regression",
        "Decision Trees",
        "Random Forest",
        "Feature Engineering",
        "Cross-validation",
        "Model Evaluation",
        "Statistical Analysis",
        "Hypothesis Testing",
        "Data Preprocessing",
        "Correlation Analysis",
        "Time Series Analysis",
        "K-fold",
        "Data Aggregation",
        "Data Importing",
        "Data Manipulation",
        "Pearson Correlation",
        "Regression Analysis",
        "Polynomial Regression",
        "Learning Curve Analysis",
        "r2_score",
        "mean_squared_error",
        "Classification Metrics",
        "confusion_matrix",
        "classification_report",
        "ANOVA testing",
        "t-test",
        "Z-score Standardization",
        "Statistical Hypothesis Testing",
        "p-value Interpretation",
        "Data Transformation",
        "Model Validation",
        "Statistical Significance Testing","Data Scaling",
        "Feature Scaling",
        "Categorical Data Encoding",
        "Feature Selection",
        "Model Analysis",
        "Hyperparameter Tuning",
        "Root Mean Squared Error",
        "Mean Absolute Error",
        "Residual Analysis for Regression Models",
        "Feature Importance Analysis",
        "Actual vs Predicted Visualization",
        "Model Interpretation",
        "Supply Chain Forecasting",
        "Business Impact Analysis",
    ]),
    ("Object Oriented Programming and Higher Order Functions | Python", [
        "Python",
        "regex",
        "Higher order functions (HOF)"
        "Object Oriented Programming (OOP)"
    ]),
    ("Ribsosome | Python", [
        "python",
        "regex", 
        "higher order functions (HOF)"
    ]),
    ("Regex Finite State Machine | Python", [
        "python",
        "regex",
        "nondeterministic finite automata (NFA)",
        "deterministic finite automata (DFA)",
        "NFA to DFA", 
        "determine if string is accepted by a regex using its finite state machine representation",
        "implemented concat of two FSMs representing regex strings",
        "implemented union of two FSMs representing regex strings",
        "implemented star (Kleene closure) of a FSM representing a regex string"
    ]),
    ("OCamel Higher Order Function Utilities | OCamel", [
        ""
        "functional programming"
        "OCamel" 
        "implemented higher order functions (HOF) in OCamel"
        "used HOF in OCamel to solve complex problems"
        "Recursion"
    ]), 
    ("Regex Finite State Machine | OCamel", [
        "OCamel",
        "functional programming",
        "Recursion",
        "Regex",
        "nondeterministic finite automata (NFA)",
        "deterministic finite automata (DFA)",
        "NFA to DFA",  
        "determine if string is accepted by a regex using its finite state machine representation",
        "implemented concat of two FSMs representing regex strings", 
        "implemented union of two FSMs representing regex strings",
        "implemented star (Kleene closure) of a FSM representing a regex string"
    ]),
    ("Lambda Calculus Interpreter | OCamel", [
        "OCamel", 
        "functional programming",
        "Recursion",
        "parsing", 
        "interpreting", 
        "lexer / lexing", 
        "lambda calculus"
    ]),
    ("Rust Utilities | Rust", [
        "Rust",
        "Regex"
    ]),
    ("Reddit Real-time Sentiment Analyzer | Python, Flask, React, Docker, Machine Learning", [
        "python",
        "reddit API", 
        "docker",
        "REST api",
        "toml file for dependencies", 
        "javascript",
        "bootstrap",
        "react",
        "sentiment analysis", 
        "JSON",
        "HTTP requests",
        "reddit data is consumed into pandas data frames",
        "Leveraged OpenAI’s API to access ChatGPT which could provide the user with a brief summary of the topic",
        "flair sentiment analysis",
        "roBERTo sentiment analysis",
        "vader (NLTK) sentiment analysis",
        "flask",
        "collaborated with other talented software engineers (team of 5)",
        "machine learning" 
    ]),
    ("Image Classification Using Convolutional Neural Networks | Python, Pytorch, Pandas, Machine Learning", [
        "python",
        "Convolutional Neural Network", 
        "Collaborated (team of 2)", 
        "Model design", 
        "matplot lib", 
        "pandas",
        "pytorch",
        "image classifcation", 
        "machine learning model validation", 
        "machine learning model evaluation",
        "machine learning"
    ]),
    ("Reinforcement Learning Pathing | Python, NumPy, Artificial Intelligence", [
        "python",
        "reinforcment learning", 
        "q learning",
        "numpy", 
        "pygame",
        "gymnasium",
        "cv2",
        "Artificial Intelligence AI" 
    ]),
    ("Classification Using Supervised Learning | Python, Matplotlib, Machine Learning", [
        "decision tree" 
        "nearest neighbor", 
        "perceptron",
        "collaborated with other talented software engineers (team of 3)",
        "numpy",
        "matplotlib", 
        "pylab",
        "python",
        "machine learning"
    ]),
    ("Multiclass Classification and Linear Models | Python, NumPy", [
        "python",
        "numpy",
        "binary classification",
        "gradient descent",
        "loss function implementation",
        "perceptron  / linear classifier",
        "multiclass classification",
        "machine learning"
    ]),
    ("PCA, SoftMax, and Neural Networks | Python, NumPy, Matplotlib, SciPy", [
        "python",
        "collaborated with another talented software engineer (team of 1)",
        "Neural Network", 
        "PCA",
        "dimensionality reduction", 
        "Softmax regression",
        "numpy", 
        "gradient descent", 
        "scipy",
        "pylab",
        "matplot lib"
    ]),
    ("PCA and RANSAC | Python, Matplotlib, NumpPy", [
        "PCA",
        "Ransac",
        "matplotlib",
        "numpy", 
        "python",
    ]),
    ("Color Segmentation using Gaussian Mixture Model | Python, cv2, NumPy, Matplotlib", [
        "computer vision", 
        "color segmentation using Gaussian Mixture Model",
        "matplotlib",
        "cv2", 
        "numpy",
        "object detection using color segmentation",
        "object depth measurement", 
        "collaborated with a team of 4 software engineers", 
        "python",
        "documented findings in a final report"
    ]),
    ("Panorama Stitching | Python, cv2, NumPy, Matplotlib", [
        "computer vision", 
        "panorama stitching", 
        "matplotlib", 
        "cv2",
        "corner detection", 
        "Adaptive Non-Maximal Suppression (ANMS)",
        "numpy",
        "created 40x40 feature descriptor for each corner",
        "match features using the sum of squared differences between feature descriptors",
        "utilized RANSAC while simultaneously computing a homography matrix to remove outlier matches and find the best homography matrix",
        "Project, warp, and blend images so that corresponding feautres overlap and it forms a seamles larger image",
        "collaborated with a team of 5 software engineers",
        "python",
        "documented findings in a final report"
    ]),
    ("3D Scene Reconstruction Using Two Views | Python, cv2, NumPy, Matplotlib", [
        "python",
        "numpy", 
        "cv2", 
        "matplotlib", 
        "computer vision", 
        "feature matching", 
        "Leveraged RANSAC to estimate the fundamental matrix", 
        "Epipolar Lines",
        "Estimated the Essential Matrix",
        "extracted Camera Poses using the Essential Matrix",
        "Utilized linear triangulation to determine the relative position of points in the image",
        "Cheriality Condition",
        "Leveraged COLMAP to perform 3D scene reconstruction using many images",
        "documented findings in a final report",
        "collaborated with a team of 5 software engineers"
    ]),
    ("Depth Estimation Using Stereo Vision | Python, cv2, NumPy, Matplotlib", [
        ""
        "Computer vision",
        "python",
        "numpy", 
        "depth estimation using stereo vision",
        "image projection",
        "computed image disaprity", 
        "displaying disaprity map using matplotlib",
        "implemented and leveraged sum of squared difference (SDD), sum of absolute difference (SAD), and zero-normalized cross correlation (ZNCC) image kernels in disaprity calculation",
        "cv2",
        "collaborated with a team of 5 software engineers", 
        "matplotlib"
    ]),
    ("Racket Compiler: Conditionals | Racket, C, x86 Assembly", [
        "compiler", 
        "interpreter",
        "parser", 
        "ast", 
        "racket", 
        "x86",
        "c runtime",
        "lldb debugging of x86 code",
        "functional programming",
        "software testing",
        "drracket",
        "literals",
        "unariy operators",
        "add1",
        "sub1",
        "abs",
        "negate",
        "zero?",
        "N-ary operators",
        "case", 
        "cond", 
        "primitives" 
    ]),
    ("Racket Compiler: Many Variables | Racket, C, x86 Assembly", [
        "compiler", 
        "interpreter", 
        "parser", 
        "ast", 
        "racket", 
        "x86", 
        "c runtime", 
        "lldb debugging of x86 code",
        "functional programming",
        "software testing",
        "drracket", 
        "cond", 
        "case",
        "let*", 
        "let",
        "- (negate)",
        "abs",
        "not",
        "integer?",
        "boolean?",
        "interpreter environment to keep track of variable bindings",  
        "compiler closure to keep track of variable offsets at compile time"
    ]),
    ("Racket Compiler: Pattern Matching | Racket, C, x86 Assembly", [
        "Built a Racket-to-x86 Compiler integrating x86 Assembly, Racket, and C runtime, implementing stack-based calling conventions and tail-call optimization. Achieved a 40% reduction in recursive function overhead based on benchmark performance tests.",
        "Resolved 15+ segmentation faults in memory pointer operations by debugging x86 instruction traces using GDB/LLDB. Utilized systematic watchpoint analysis to identify and correct pointer dereferencing and memory allocation issues.",
        "Utilized Test-Driven Development (TDD) with the RackUnit framework to guide compiler development, applying functional programming principles. Wrote 150+ test cases covering 15+ Racket language features (e.g., arithmetic, I/O, pattern matching), which ensured correctness and reduced logic errors by 95% through thorough edge-case coverage.",
        "interpreter", 
        "parser",
        "ast",
        "software testing",
        "drracket",
        "pattern matching",
        "pattern matching of lists",
        "pattern matching of vectors",
        "software testing"
    ]),
    ("Racket Compiler: Error Handling | Racket, C, x86 Assembly", [
        "compiler",
        "interpreter",
        "parser",
        "ast",
        "racket",
        "x86",
        "c runtime",
        "lldb debugging of x86 code",
        "functional programming",
        "software testing",
        "drracket",
        "error handling", 
        "artiry check",
        "viardic functions", 
        "apply (apply a function to a list of arguments with the last argument being a list of args to “decompose” into individual arguments)"
    ]),
    ("PPMI Word Embeddings | Python, NumPy", [
        "python", 
        "numpy",
        "natural language processing",
        "word embeddings",
        "dense word embeddings", 
        "sparse word embeddings", 
        "utilized pickle to cache the embedding model",
        "pointwise mutual information (PMI)",
        "positive pointwise mutual information (PPMI)",
        "cosine similarity",
        "distributional semantics"
    ]),
    ("OpenMP Parallelization | C++, OpenMP", [
        "c++", 
        "openMp",
        "threads",
        "report documenting findings",
        "performance analyis",
        "executing jobs against a parallel cluster",
        "ssh into remote cluster",
        "Zaratan high performance cluster",
        "linux",
        "command line interface",
        "nano",
        "OpenMP (directive-based parallelism)",
        "Loop Parallelization (directive implementation)",
        "Thread Scaling (1-64 threads)",
        "Performance Optimization",
        "Strong Scaling Analysis",
        "C++ Programming",
        "Makefile Build Systems",
        "Compiler Flags (-fopenmp, -O2)",
        "Correctness Verification",
        "Batch Processing (SLURM scripts)",
        "omp_get_wtime() (OpenMP timing)",
        "Execution Profiling",
        "Runtime Comparison",
        "Resource Binding (--exclusive, --mem-bind)",
        "Processor Affinity (OMP_PROCESSOR_BIND)",
        "Computational Geometry (closest points)",
        "Graph Algorithms (adjacency matrix)",
        "Vector Processing (element-wise operations)",
        "Signal Processing (Discrete Fourier Transform)",
        "Numerical Computation",
        "Zaratan HPC Cluster",
        "Node-Level Parallelism",
        "Thread Affinity Control",
        "GCC Toolchain (v11.3.0)",
        "Job Scheduling",
        "Deterministic Parallelism",
        "Random Seed Control",
        "Problem Sizing (16384, 8192, etc.)",
        "Correctness Testing",
        "Output Validation"
    ]), 
    ("MPI Game of Life | C++, MPI", [
        "c++",
        "game of life",
        "processes",
        "race conditions",
        "synchronization",
        "blocking communication between threads to communicate edge rows/cols",
        "non blocking communication between threads to communicate edge rows/cols",
        "MPI_Gather together peices of the board",
        "MPI_Scatter to distribute peices of the game board",
        "report documenting findings",
        "performance analyis",
        "executing jobs against a parallel cluster",
        "ssh into remote cluster",
        "Zaratan high performance cluster",
        "linux",
        "command line interface",
        "nano",
        "Correctness Verification",
        "Performance Profiling",
        "Statistical Timing Metrics (min/avg/max)",
        "Scalability Plotting",
        "Cellular Automata (Game of Life)",
        "Finite Board Simulation",
        "Conway's Rules Implementation",
        "Distributed Data Initialization",
        "Edge Case Handling (non-wraparound boundaries)",
        "OpenMPI Implementation",
        "Zaratan HPC Environment",
        "Cluster Computing",
        "Strong Scaling Analysis",
        "C/C++ Development",
        "Makefile Build Systems",
        "Batch Job Processing (HPC cluster scheduling)",
        "CSV Data Handling",
        "Memory Contiguity Management",
        "Execution Timing (MPI_Wtime)",
        "Performance Scaling (strong scaling tests)",
        "Reduction Operations (MPI_MIN/MPI_MAX/MPI_SUM)",
        "Compiler Optimization (-O2 flags)",
        "MPI Programming (Message Passing Interface)",
        "Non-blocking Communication (MPI_Isend/MPI_Irecv)",
        "Domain Decomposition (1D row-based partitioning)",
        "Ghost Cell Exchange",
        "Load Balancing (static distribution)",
        "Parallel Algorithm Design"
    ]),
    ("Parallel Performance Analysis Tools | Python, OpenMP, MPI, HPCToolkit, Caliper, Hatchet", [
        "report documenting findings",
        "performance analyis",
        "executing jobs against a parallel cluster",
        "ssh into remote cluster",
        "Zaratan high performance cluster",
        "linux",
        "command line interface", 
        "nano",
        "HPCToolkit (performance profiling)",
        "Caliper (performance measurement)",
        "Hatchet (performance data analysis)",
        "MPI (Message Passing Interface)",
        "OpenMPI (MPI implementation)",
        "Parallel Program Optimization",
        "Performance Analysis",
        "Load Imbalance Analysis",
        "Profiling (hpcrun/hpcstruct/hpcprof)",
        "High-Performance Computing (HPC)",
        "Programming & Scripting",
        "Python (analysis scripts)",
        "C++ (LULESH codebase)",
        "Bash Scripting (cluster job execution)",
        "Linux Command Line",
        "Git (version control)",
        "HPC Cluster Workflows",
        "Batch Job Processing (cluster scheduling)",
        "Resource Allocation (core/process management)",
        "CMake (build systems)",
        "GCC Compiler (v9.4.0)",
        "MPI Execution (mpirun)",
        "Data Analysis & Visualization",
        "Performance Metrics (exclusive time)",
        "Dataframe Operations (Hatchet/pandas)",
        "JSON Data Processing",
        "Statistical Analysis (time differences)",
        "Exclusive Time Measurement",
        "Scaling Analysis (1/8/27/64-core comparisons)",
        "Function-Level Optimization",
        "Graphframe Manipulation (Hatchet)",
        "Cluster Computing (zaratan HPC)"
    ]),
    ("Cuda Game of Life | C++, Cuda", [
        "wrote a report documenting findings",
        "performance analyis",
        "executing jobs against a parallel cluster",
        "ssh into remote cluster",
        "Zaratan high performance cluster",
        "linux",
        "command line interface",
        "nano",
        "CUDA Development",
        "GPU Kernel Design (compute_on_gpu, padded_matrix_copy)",
        "Thread Striding",
        "Block/Grid Configuration (8x8 blocks, command-line grid sizing)",
        "Host-Device Memory Transfer (copy_grid_to_device, copy_grid_to_host)",
        "nvcc Compiler",
        "GPU Timing Metrics (millisecond precision)",
        "Block Size Optimization",
        "Performance Scaling Analysis",
        "Kernel Optimization",
        "Resource Binding (A100 GPU)",
        "Cellular Automata Implementation",
        "Conway's Game of Life",
        "Finite Board Simulation",
        "Game of Life Computation",
        "Memory Padding (padded_matrix_copy)",
        "CSV Data Handling",
        "Zaratan HPC Cluster",
        "SLURM Job Scheduling",
        "GPU Resource Allocation (--gres=gpu:a100_1g.5gb)",
        "Batch job Processing",
        "Birth/Survival/Death Rules for game of life",
        "State Transition Logic",
        "Strided Parallelism",
        "Output Sorting (X/Y coordinate ordering)",
        "Grid Decomposition",
        "C++"
    ]),

    ("Cuda Video Kernels | C++, Cuda", [
        "report documenting findings",
        "performance analyis",
        "executing jobs against a parallel cluster",
        "ssh into remote cluster",
        "Zaratan high performance cluster",
        "linux",
        "command line interface",
        "nano",
        "CUDA",
        "GPU Kernel Development",
        "Image Convolution",
        "Video Processing",
        "Computer Vision",
        "Image Kernels",
        "Blur Effect",
        "Edge Detection",
        "Sharpen Effect",
        "Identity Kernel",
        "BGR Pixel Format",
        "Kernel Convolution",
        "GPU Striding",
        "Host-Device Memory Transfer",
        "OpenCV",
        "nvcc Compiler",
        "Makefile",
        "GPU Timing",
        "Frames Per Second (FPS)",
        "Performance Optimization",
        "Block Size Configuration",
        "Grid Configuration",
        "NVIDIA A100 GPU",
        "Zaratan HPC",
        "SLURM Job Scheduling",
        "GPU Resource Allocation",
        "Video Frame Processing",
        "Border Handling",
        "Convolution Algorithm",
        "GPU Performance Tuning",
        "Image Effects",
        "Video Filtering",
        "GPU Kernel Optimization",
        "HPC Environment",
        "GPU Acceleration",
        "C++"
    ]),
    ("Simulated CPU Running ARM | C, ARM Assembly", [
        "C",
        "x86",
        "hardware",
        "manipulate registers",
        "supports ADD, SUB, AND, ORR, EOR, LSL, LSR, ASR, LDR, STR, CMP, MOV, B, BEQ, and special END instructions",
        "manipulate program counter"
    ]),
    ("Graphical Town Explorer | Java, JUnit, JavaFX", [
        "Applied Test-Driven Development (TDD) and implemented quality assurance protocols using JUnit 5 Jupiter in a Java-based graph routing application. Designed 36 test cases covering Dijkstra's algorithm, edge validation, and exception handling, achieving 100% critical path coverage and reducing graph operation defects by 75%.",
        "Designed and implemented a maintainable Java object-oriented programming (OOP) solution for graphical town mapping software by developing custom Town, Road, and Graph classes using encapsulation and inheritance. Solved complex pathfinding challenges using Dijkstra’s algorithm, processing 100+ node networks with 40% faster runtime compared to the adjacency matrix approach.",
        "Engineered a modular and maintainable Java solution for a town routing system, featuring robust error handling (25+ edge cases), a SOLID-compliant class hierarchy, and full graph CRUD operations. Scaled to 100+ nodes while maintaining 100% specification compliance.",
        "Java",
        "JUnit",
        "unit testing",
        "construct graph data structure based on user input to represent cities and connecting roads",
        "roads have a source, a destination, and a distance",
        "includes add, edit, delete operations to edit roads and destinations in the map",
        "implemented dijkstra’s shortest path algorithm to find the best route between two cities in terms of distance",
        "object oriented programming OOP",
        "interfaces",
        "inheritance",
        "javafx for gui",
    ])
] 
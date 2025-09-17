from haystack.components.embedders import SentenceTransformersTextEmbedder
import numpy as np 
import os 
import fitz
from collections import defaultdict

PROMMUNI = True
PROJECTS = False

#Goal:
    #based on the sentence embedding of a list of skills, argsort the following resume items 
        #related course experience
        #work experience bullet points 
        #projects & corresponding bullet points


# Setup text embedder
text_embedder = SentenceTransformersTextEmbedder(model='all-mpnet-base-v2', progress_bar=False)
text_embedder.warm_up() 

# Embedd the list of skills
skills_result = text_embedder.run("""
Software development
Agile development
Object-oriented programming
Java
C++
C#
Python
JavaScript
TypeScript
HTML
CSS
Front-end development
Angular
Web-based programming
Test-driven development (TDD)
Unit testing
Test automation
Database interaction
MongoDB
Oracle DB
SQL Server
PostgreSQL
SQLite
Algorithms
Data structures
Dependency package managers
NPM
Gradle
Conan
Version control
Git
GitHub
GitHub Advanced Security
CodeQL
Static code analysis
User interface design
User interaction feedback
Code reviews
Design reviews
Collaboration with architects and stakeholders
Troubleshooting software malfunctions
Time management
Self-motivation
Mentorship and training of team members
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
    "ENGL393 - Technical Writing",
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
best_courses = best_courses[-10:]
# print best resumes w/ cosine similarity score
for i in range(len(best_courses)-1, -1, -1):
    print(course_similarity[best_courses[i]], courses[best_courses[i]][:112])
# Formatting for output
print("================================================Courses END================================================")
print("")


if PROMMUNI:
    # List of work experience bullet points
    work_experience = [
        # UPDATED AS OF 9/14/25
        "built a recommendation engine",
        "built front end components in react", 
        "supported backend data pipeline in postgressSQL",
        "orchastrated development using nextJS",
        "Served as the primary advisor on AI/ML capabilities at an early-stage startup, guiding technical strategy on recommendation systems and AI/ML pipelines.",
        "Designed and implemented a k-Nearest Neighbors (kNN) roommate recommendation engine, using Python (NumPy, Pandas, scikit-learn) and text embeddings to match users based on categorical data and free-text profiles.",
        "Refactored recommendation engine into an object-oriented architecture, implementing update and fetch functions, caching strategies, and performance benchmarking (100–10,000+ users).",
        "Conducted empirical testing of recommendation performance, optimizing cosine similarity computations, KDTree indexing, and filter logic to balance accuracy and efficiency.",
        "Explored vector databases (Milvus, ChromaDB) for storing and retrieving embeddings to support scalable AI-driven matchmaking.",
        "Built data generation and preprocessing pipelines (Python, Pandas, Faker, NumPy) to create synthetic datasets aligned with internal data policies for model development.",
        "Researched self-supervised and alternative ML models (cosine similarity, gradient boosting, constraint satisfaction, LLMs) to evaluate tenant scoring and roommate compatibility.",
        "Prototyped an AI-powered profile summarizer using LLMs and retrieval-augmented generation (RAG) pipelines to generate structured insights from unstructured user data.",
        "Investigated Supabase (Postgres) integration with AI/ML services, analyzing schema consistency, edge function triggers, and query-time filtering for recommendation delivery.",
        "Collaborated in a fast-paced startup environment with weekly whiteboarding sessions, Jira-based task management, and cross-functional syncs with founders and engineers.",
        "Set up and automated workflows with n8n (low-code platform) for a smart leasing assistant, integrating APIs, Docker deployment, and webhook triggers.",
        "Developed front-end components for the production web application using Next.js, React, and TypeScript.",
        "Implemented state management, hooks, and reusable components in React to enhance interactivity and scalability of the housing platform.",
        "Coordinated with team leads to translate product requirements into AI/ML and full-stack engineering tasks, including features for user onboarding, roommate matching, and tenant scoring.",
        "Documented and tracked development progress via GitHub issues, Jira tickets, and Slack, ensuring transparent communication and alignment across the team.",
        "Benchmarked multiple recommendation algorithms (constraint satisfaction, gradient boosting, kNN, cosine similarity, LLM-based) to determine tradeoffs in speed, scalability, and user experience.",
        "Tested model scalability by running recommendation algorithms on datasets ranging from 100 to 10,000+ users to measure performance under realistic growth scenarios.",
        "Researched schema design and database integration strategies for Supabase to align AI/ML recommendations with backend data models.",
        "Collaborated directly with CTO and product team to shape core recommendation system architecture and its integration into the housing platform.",
        "Investigated API design patterns to streamline how recommendations are requested, updated, and delivered across the platform.",
        "Performed debugging and issue resolution across front-end and back-end systems, including resolving data schema mismatches and integration failures.",
        "Explored caching strategies to reduce latency and computational overhead in recommendation queries.",
        "Developed modular code for embeddings and similarity functions to improve maintainability and enable rapid experimentation with new models.",
        "Analyzed and documented system tradeoffs (e.g., SQL vs vector databases, kNN vs boosting, LLM integration vs traditional ML) to support executive decision-making.",
        "Enhanced collaboration efficiency by documenting workflows, test results, and technical tradeoffs for non-technical stakeholders.",
        "Proposed monetization strategies for the housing platform, including lead-generation to landlords and a “pod compatibility” score to sell higher-quality pods.", 
        "Prototyped an LLM integration: attempted to connect a local Ollama instance with DeepSeek R1 to prototype a user profile summarization pipeline and identified API / cost blockers.", 
        "Ran the local development instance to audit front-end data capture and translate user attributes into features for machine learning models.",
        "Discovered and documented front-end UI bugs (e.g., username validity not rechecked without refresh; negative minimum budget allowed; max budget < min budget allowed; inability to edit lease length in roommate preferences).",
        "Investigated Supabase database access (used DATABASE_URL / Beekeeper Studio to inspect tables) and flagged inconsistent / messy values in the DB schema.",
        "Implemented preprocessing code to convert dates into “days since” a reference date, processed the combined_text column (converted each row’s combined text into a text embedding and appended it to the row), and changed the pipeline to return a NumPy matrix for nearest-neighbor search using cosine similarity metric.",
        "Converted the recommendation logic into a class-based design (private helper methods, stored sorted-recommendation dictionary, fit and query methods) and later moved the recommendation class, data generation, and testing code into separate files for maintainability.",
        "Designed and ran empirical performance tests and quality checks for the recommender.", 
        "Implemented and integrated query-time filtering of recommendations so users can adjust recommendation filters on the fly.",
        "Benchmarked nearest-neighbor options (KDTree vs sklearn KDTree vs scipy cKDTree), researched normalization/standardization needs (min-max vs z-score), and assessed partial-update limitations of KDTree for the use case.", 
        "Evaluated caching / precomputation deployment options for recommendations (Supabase Edge Functions vs a periodic Flask job + API) to reduce recomputation overhead.",
        "Reviewed a previous team branch (manual user-pair score vs learned model), assessed its assumptions, and recommended using cosine-similarity as a pragmatic approach until a labeled data pipeline exists.",
        "Measured and quantified the contribution of the word-embedding portion of user vectors to total similarity, informing feature-engineering decisions.",
        # "Contributed to containerized deployment setups (Docker) for AI/ML workflows and low-code automation tools (n8n).",
        # "Created reusable prompts and structured pipelines for LLM evaluation, testing prompt engineering techniques to improve reliability of AI-generated outputs."
        # "Contributed to front-end UI/UX development with Radix UI primitives, accessibility-focused popover components, and ESLint/Prettier for code quality.",
        # "Implemented front-end accessibility improvements using Radix UI and best practices in semantic React components.",
       
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
    print("================================================Prommuni Experience START================================================")
    best_work_experience = best_work_experience[-20:]
    # print best resumes w/ cosine similarity score
    for i in range(len(best_work_experience)-1, -1, -1):
        print(work_experience_similarity[best_work_experience[i]], work_experience[best_work_experience[i]])
        print()
    # Formatting for output
    print("================================================Prommuni Experience END================================================")
    print("")


# List of work experience bullet points
work_experience = [
    "Logged changes in the knowledge graph and displayed these changes on the admin dashboard for review by HR leaders.",
    "Created user authentication page using popular password encryption libraries.",
    "Leveraged sharepoint.",
    "Integrated OpenAI’s API into a retrieval-augmented generation pipeline for question answering, using Jinja templates to engineer few-shot prompts that combined explicit task instructions with example outputs and the most relevant knowledge graph subsets retrieved via vector embedding similarity, resulting in an 80% improvement in response accuracy.",
    "Built a retrieval-augmented generation (RAG) pipeline that empowered HR professionals to ask high-level, natural language questions against a structured knowledge graph, leveraging OpenAI’s API and Jinja-based prompt engineering to deliver highly accurate responses, improving overall decision-making efficiency.",
    "Utilized ChatGPT and DeepSeek to resolve complex development challenges, reducing implementation time and improving operational efficiency.",
    "Utilized Secure File Transfer Protocol (SFTP) to send files to the AWS instance hosting our web application.",
    "Implemented 50+ REST API endpoints for handling JSON payloads and HTTP requests, reducing CRUD operation time by 30% through dynamic AJAX components and modular design patterns.",
    "Drove full-stack development of an internal HR analytics platform across the full Software Development Life Cycle (SDLC), from requirements gathering and architecture design to testing and deployment, delivering production-ready features in Agile sprints and enhancing HR insights accessibility by 40%.",
    "Built a consumer-facing HR analytics dashboard using JavaScript and Chart.js, translating Neo4j knowledge graph CRUD operations into real-time time-series visualizations. Reduced HR decision latency by 40% and improved anomaly detection by 65% through continuous change monitoring.",
    "Led product research and requirements analysis by evaluating Obsidian’s features and workflows, translating insights into system design and Neo4j schema architecture. This approach enabled the full-stack team to reduce development time by 25% and deliver an internal talent-tracking web app more efficiently.",
    "Engineered retrieval-augmented generation (RAG) pipelines for our HR knowledge graph application, implementing bidirectional relationship extraction and token optimization strategies that improved context relevance by 50% while troubleshooting GPU acceleration to reduce model inference latency by 30%.",
    "Programmed Arduino microcontrollers in C/C++ to design and debug digital circuits (button-controlled LED matrices, sensor-triggered fans), implementing GPIO manipulation and interrupt handling to advance foundational hardware and embedded systems knowledge.",
    "Led Docker containerization and deployment of an internal HR analytics platform to AWS, collaborating with engineers, cybersecurity specialists, and C-suite stakeholders to improve availability and enable scalable production rollout.",
    "Deployed a containerized application to AWS using Docker on a Linux instance, resolving port conflicts over TCP/IP and accelerating future deployments by 40%. Simulated the production environment locally with VMware and VirtualBox, and managed AWS servers via CLI tools.", #= to prior
    "Delivered maintainable Neo4j CRUD operations to manage 100+ employee records, enhancing backend data handling and platform scalability.", 
    "Developed knowledge graph embeddings using Neo4j's vector indexing capabilities, leveraging cosine similarity search that improved node relationship discovery accuracy by 55% for our AI-powered HR analytics web application.",
    "Automated JSON data ingestion handling 100+ daily HTTP requests, integrating schema validation to reduce parsing errors by 65%. Connected the pipeline to a RESTful API built with Python and Flask for real-time updates to the Neo4j database.",
    "Designed and built responsive UI components with Bootstrap and jQuery, integrating asynchronous HTTP handlers to efficiently render backend JSON payloads and reduce load times by 35% in large-scale knowledge graph visualizations.",
    "Developed and configured Neo4j vector search indexes to support graph embeddings, enabling semantic similarity comparisons that enhanced response accuracy in the question-answering RAG (Retrieval-Augmented Generation) ML pipeline by 75%.",
    "Engineered HR knowledge graph solutions using Neo4j and Cypher queries, collaborating with three software engineers to implement graph-based search and Natural Language Processing capabilities that reduced data retrieval time by 65%",
    "Developed Python/Flask RESTful API endpoints with OpenAI integration for resume parsing, using prompt engineering and Jinja templates to extract structured data including skills, education, and certifications from resumes, increasing extraction accuracy by 40% and significantly reducing manual input time.",
    "Engineered a full-stack knowledge graph application with a Python/Flask backend and JavaScript/Bootstrap frontend, implementing dynamic AJAX/jQuery UI components and real-time form validation that cut CRUD operation time by 30%.", #= to next
    "Engineered a full-stack knowledge graph application using Python/Flask backend and JavaScript/Bootstrap frontend, collaborating with 3 software engineers to implement 15+ REST API endpoints for CRUD operations on Neo4j graph databases, reducing data retrieval latency by 40% through optimized Cypher queries.", #= to prior
    "Collaborated closely with a team of three Software Engineers to build a full-stack web application using Python and Flask for the backend, and HTML, CSS, JavaScript, and Bootstrap for the front end, while fostering effective communication that accelerated development and enabled delivery two weeks ahead of schedule.",
    "Developed a flexible data ingestion pipeline in Python using Flask for API integration and Pandas for data processing, enabling quick graph node creation from search bar input or CSV upload via a custom adjacency list syntax, cutting data loading complexity by 70%",
    "Revamped user interfaces with Bootstrap/CSS, implementing modal-based editing and autocomplete search that decreased user errors by 45% in node-relationship management workflows.", 
    "Integrated OpenAI API for question answering, retrieval-augmented generation, leveraged Jinja to perform prompt engineering, which improved response accuracy by 80% while implementing usage caps to control API costs.",
    "Created comprehensive repository wiki documentation for graph schemas and RESTful API endpoints, improving onboarding efficiency by 50% for new developers while standardizing data ingestion protocols.",
    "Developed an elegant graph view using HTML5, CSS, JavaScript, Bootstrap, and Alchemy.js while using AJAX and RESTful API calls for dynamic updates, enabling real-time graph updates that accelerated HR data visualization by 90%.",
    "Implemented schema validation with Python, Flask, and Neo4j, creating node-type checking algorithms that reduced data inconsistencies by 80% in knowledge graph updates.",
    "Refactored 5,000+ lines of Python, HTML, and JavaScript code in an Agile team setting using Git, resolving 15+ critical bugs and improving maintainability and reliability by 90%.", 
    "Prototyped local LLM solutions using Hugging Face models and vector embeddings, enabling retrieval-augmented generation in a knowledge graph application and achieving 50% faster candidate search queries.",
    "Evaluated Hugging Face models and vector embeddings to integrate retrieval-augmented generation into a knowledge graph platform, delivering prototypes that accelerated candidate search query resolution by 50%.",
    "Designed responsive user interfaces with Bootstrap/Jinja templates, implementing dynamic layouts by leveraging AJAX and jQuery to modify page layouts without necessitating page refresh, resulting in a more satisfying user experience.",
    "Developed interactive relationship visualizations with Alchemy.js that enabled users to intuitively input and link personnel data, reducing user errors by 25%.",
    "Visualized Human Resource information highlighting skills, certifications, and clearances held by the internal workforce using HTML5, CSS, JavaScript, and Alchemy.js, producing relationship visualization components that decreased user errors by 25% in graph node and relationship creation workflows.",
    "Developed knowledge graph-based search and Natural Language Processing features including autocomplete functionality using Cypher query language, enabling complex relationship mapping across 200+ nodes which improved HR data query efficiency by 60%",
    "Integrated Python/Flask RESTful API endpoints with a JavaScript front end using AJAX and jQuery, developing dynamic node/relationship management interfaces that reduced CRUD operation time by 30% through real-time form validation and modal-based editing.",
    "Containerized an HR analytics platform using Docker to enable scalable deployment on AWS, streamlining the CI/CD pipeline and reducing deployment errors by 30%.",
    "Documented 20+ RESTful API endpoints and internal code functionality, accelerating onboarding speed by 50% for new software engineering hires.",
    "Created data ingestion systems using Pandas to process CSV data into Neo4j knowledge graphs, streamlining the ingestion of over 200+ nodes and 100+ relationships.",
    "Led front-end bug resolution and feature enhancements in an Agile environment using Git and GitHub, refactoring 5K+ lines of front end JavaScript, HTML5, and CSS code into modular components that improved maintainability by 50%.", 
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
print("================================================ATS Experience START================================================")
best_work_experience = best_work_experience[-20:]
# print best resumes w/ cosine similarity score
for i in range(len(best_work_experience)-1, -1, -1):
    print(work_experience_similarity[best_work_experience[i]], work_experience[best_work_experience[i]])
    print()
# Formatting for output
print("================================================ATS Experience END================================================")
print("")

if PROJECTS: 
    # For projects we use a list of tuple (projectTitle, [bullet points])
        #but we embedd projectTitle+bulletPoint for each bullet point to find most relevant project experience
    projects = [
        ("SQL Data Analysis | SQL, Pyhon", [
            "Executed complex SQL queries to extract and analyze employee compensation data, applying data filtering, sorting, and summarization techniques to uncover key workforce insights.",
            "Utilized SQL aggregation functions to compute summary statistics such as average pay for each position, enhancing understanding of compensation trends.",
            "Executed complex multi-clause SQL queries for compensation analysis across 100k+ records, incorporating filtering, sorting, and aggregation to identify compensation trends.", 
            "Queried 100K+ rows of employee data using SQL (SELECT, WHERE, ORDER BY) to identify high-earning employees hired in 2012, sorted by descending salary.",
            "Loaded SQL query results into Pandas DataFrames for further transformation and analysis in Python, enabling scalable processing of 100K+ employee records.",
            "Used SQL (SELECT, GROUP BY, AVG(), ROUND(), AS) to calculate and rename average compensation by hire year, streamlining insights from 100K+ employee records.",
            "Generated detailed compensation summaries per job title by using SQL aggregation functions (AVG, ROUND) and column aliasing on 100K+ observations to uncover salary, benefits, overtime, and total compensation averages.",
            "Automated the extraction and partitioning of employee salary data by hire year using SQL queries and Python loops; utilized pandas.read_sql() and pandas.to_sql() to create year-specific tables from a dataset of 100K+ records.",
            "Performed SQL INNER JOIN on director_id to combine 7K+ rows of movie and director data, enriching the dataset for relational insights.",
            "Built an aggregated analysis to uncover the top 5 directors by average IMDB rating using SQL (INNER JOIN, GROUP BY, ORDER BY, AVG, LIMIT) on 7K+ movie records.",
            "Identified the top 5 directors by average movie budget through SQL queries involving joins, grouping, sorting, and limiting functions on a dataset of 7K+ entries.",
            "Applied advanced SQL functions in SQLite to execute multi-clause queries involving filtering, sorting, grouping, and aggregation across 100K+ records, delivering actionable insights from large Kaggle datasets.",
            "Performed rigorous data analysis and exploration on employee compensation and movie datasets by using SQL GROUP BY, AVG, ROUND, and conditional logic to surface key trends and outliers.", 
            "Engineered SQL-based data aggregation pipelines to generate summary statistics (e.g., average salary by position and hire year), enabling better understanding of workforce and industry-level metrics.",
            "Joined multiple relational tables with INNER JOIN on foreign keys (e.g., director_id) to enrich datasets with hierarchical metadata, supporting advanced aggregation like top-5 rankings by average rating and budget.",
            "Built automated SQL workflows using Python loops and Pandas to extract, process, and organize 100K+ rows of salary data, showcasing effective use of SQL table design and data handling in Python.",
            "Leveraged SQL in a SQLite environment to process and analyze large-scale Kaggle datasets, demonstrating strong command over relational data modeling, query optimization, and real-world dataset manipulation."
            "Designed and implemented custom SQL tables to manage and analyze large-scale datasets, using Python and Pandas to support efficient data storage and querying across 100K+ records."
            "Explored and analyzed 100K+ records of employee and movie data, using multi-clause SQL queries (WHERE, GROUP BY, ORDER BY, AVG, ROUND) to identify compensation trends and top-performing directors, demonstrating strong data exploration capabilities.",
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
            "python was used as the backend language, combined with flask it served as our backend service and the host for our sentiment analysis",
            "leveraged the reddit API to get current conversations about a user specified topic for sentiment analysis", 
            "utilized docker to streamline deployment to AWS",
            "created a RESTful api for seamless communication with the react front end",
            "utilized a toml file to document dependencies", 
            "utilized javascript and react to build an elegant front end",
            "leveraged bootstrap on the front end to quickly style components ",
            "Utilized reacts state management and asynchornous operations to make our website more dynamic",
            "performed sentiment analysis on fresh reddit data to give the user an idea of the public sentiment on a given topic", 
            "Handled JSON payloads from the reddit API, extracting conversations for sentiment analysis",
            "Interacted with the backend and the reddit API via HTTP requests",
            "reddit data was consumed into pandas data frames to improve backend processing",
            "Leveraged OpenAI’s API to access ChatGPT which could provide the user with a brief summary of the topic given a subset of the discussion",
            "used flair, roBERTo, and vader (NLTK) to perform sentiment analysis on real reddit posts centered around a user specified topic",
            "Leveraged flask to build a RESTful API",
            "Collaborated in a team of 5 Software Engineers to build a full-stack web application delivering real-time Reddit sentiment analysis.",
            "used machine learning and natural language processing in the form of sentiment analysis to provide the user with insights on user sentiment.",
            "created a unified data visualization that would display sentiment as an easy to understand gauge which would dynamically update as sentiment data streams in. The data was streamed as it was ready.",
            "Developed backend using Python and Flask to perform sentiment analysis and serve processed results to the frontend via RESTful APIs.",
            "Built frontend with React, leveraging state management and asynchronous operations for dynamic user interactions.",
            "Integrated sentiment analysis models including VADER, RoBERTa, and Flair to analyze Reddit discussions on user-specified topics."
            "Designed and implemented RESTful APIs to enable seamless frontend-backend communication.",
            "Utilized the Reddit API to extract and preprocess live discussion data; handled JSON payloads and parsed relevant content.",
            "Processed and structured incoming data using Pandas for improved performance in backend computation.",
            "Leveraged OpenAI’s ChatGPT API to generate real-time summaries of Reddit topics based on filtered conversation threads.",
            "Containerized the application using Docker and deployed to AWS for scalable access and streamlined development.",
            "Created real-time data visualizations including sentiment gauges that dynamically updated as new data streamed in.",
            "Applied Bootstrap and custom CSS to design a responsive, user-friendly frontend interface.",
            "Documented dependencies and environment configuration using a TOML file for consistent team development."
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
            "Generated performance reports using Hatchet by loading and parsing Caliper-generated JSON datasets, enabling function-level execution time analysis across 1, 8, 27, and 64-core executions.",
            "Utilized Hatchet.from_caliper() to parse JSON-based performance profiles from Caliper into accessible graphframes, enabling function-level time analysis in Python.",
            "Employed pandas dataframe operations to filter and sort performance data, identifying top-N time-consuming functions across different core counts for performance analysis.",
            "Automated function-level analysis for performance bottlenecks by sorting on exclusive time and printing function names and times, improving visibility into computational hotspots.",
            "Analyzed imbalance across MPI processes by applying Hatchet’s load_imbalance() function to performance profiles obtained using HPCToolkit and sorted the results based on imbalance values using pandas.",
            "Subtracted performance graphframes between 8-core and 64-core datasets using Hatchet to compute time deltas, identifying functions with the largest time increase due to poor scalability.",
            "Developed robust sorting logic using pandas to list top-N functions with greatest exclusive time difference between 8-core execution and 64-core execution, delivering actionable insights for optimization.",
            "Used nano and Linux command line interface (CLI) tools for on-cluster code editing and configuration.",
            "Customized Python scripts to parse Caliper-generated datasets and perform statistical evaluations using pandas, enabling insights into function execution time distributions.",
            "Analyzed function-level profiling data using HPCToolkit, leveraging hpcrun, hpcstruct, and hpcprof for complete pipeline execution.",
            "Wrote cluster-compatible Bash scripts to automate job submission and environment setup, managing directory structures and software dependencies, enabling repeatable workflows for consistent SLURM job execution.",
            "Scripted performance analysis workflows in Python, using Hatchet, pandas, and JSON libraries to manipulate and visualize exclusive execution time data.",
            "Built LULESH from source using CMake, ensuring compatibility with MPI and leveraging compiler flags for optimal performance.",
            "Configured SLURM resource requests with explicit core counts and node exclusivity to minimize performance variability and support consistent parallel execution.",
            "Deployed LULESH batch jobs on the Zaratan cluster using Bash scripts, integrating mpirun and hpcrun within Slurm-compatible job scripts for automated performance data collection.",
            "Executed function-level performance differential analysis by comparing Hatchet graphframes between 8-core and 64-core runs, isolating functions with the highest exclusive time deltas.",
            "Diagnosed load imbalance across 64-core runs using Hatchet’s load_imbalance() function, pinpointing MPI routines as contributing most to inter-process delays.",
            "Conducted scaling analysis by comparing exclusive time metrics across increasing core counts, identifying diminishing returns in performance improvement beyond 27 cores.",
            "Developed Python scripts to extract and print top-N time-consuming functions using Hatchet’s dataframe API, improving visibility into computational hotspots.",
            "Remotely accessed Zaratan’s login nodes via SSH to develop performance analysis scripts directly on the HPC system, ensuring environment compatibility and fast iteration.",
            "Used Git for version control to manage and track changes across profiling scripts, supporting real-time debugging and ensuring reproducibility across high-performance computing (HPC) job executions.",
            "Troubleshot compiler errors and dependency conflicts related to MPI and OpenMP settings, ensuring a stable build environment for performance experimentation.",
            "Analyzed the C++ LULESH codebase to extract exclusive-time metrics with HPCToolkit, identifying key functions responsible for performance bottlenecks."
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

    # Create a new list where each project bullet point is prefixed with it's corresponding title
    combined_projects = [] 
    for title, bullet_list in projects: 
        for bullet in bullet_list: 
            combined_projects.append(title + " -> " + bullet)


    # Convert combined_projects into a embedding matrix 
    projects_matrix_unit = np.zeros(shape=(len(combined_projects), len(skills_embedding)))
    for i, bullet in enumerate(combined_projects):
        # Get embedding for this project bullet 
        bullet_embedding = np.array(text_embedder.run(bullet)['embedding'])

        # Normalize the embedding for cosine similarity later
        projects_matrix_unit[i] = bullet_embedding/np.linalg.norm(bullet_embedding)

    # Perform cosine similarity
    project_similarity = np.dot(projects_matrix_unit, skills_embedding_unit)

    # argsort the project bullet points by cosine similarity to provided skills 
    best_projects = np.argsort(project_similarity, axis=0)

    # Formating for output 
    print("")
    print("================================================Projects RAW START================================================")
    best_projects_short = best_projects[-20:]
    # print best resumes w/ cosine similarity score
    for i in range(len(best_projects_short)-1, -1, -1):
        print(project_similarity[best_projects_short[i]], combined_projects[best_projects_short[i]])
        print()
    # Formatting for output
    print("================================================Projects RAW END================================================")
    print("")

    #find and print best project along with it's bullet points in order of best score
    dict = defaultdict(list)
    best = None
    for i in range(len(best_projects)-1, -1, -1):
        title, bullet = combined_projects[best_projects[i]].split("->")
        dict[title].append((project_similarity[best_projects[i]], bullet))
        # the best project is the first project to get 3 bullet points when traversing project bullet point cosine similarity in descending order
        if len(dict[title]) == 3 and best is None:
            best = title

    # Formatting for output
    print("")
    print("================================================Project START================================================")
    print(best)

    # Print each bullet point in the best project along with its cosine similarity to the provided list of skills 
    for similarity, bullet in dict[best]: 
        print(similarity, bullet)
        print()

    # Formatting for output
    print("================================================Project END================================================")
    print("")

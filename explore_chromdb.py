import chromadb
import uuid


# List of work experience bullet points
work_experience = [
    "Logged changes in the knowledge graph and displayed these changes on the admin dashboard for review by HR leaders.",
    "Created user authentication page using popular password encryption libraries.",
    "Leveraged sharepoint.",
    "Integrated OpenAI’s API into a retrieval-augmented generation pipeline for question answering, using Jinja templates to engineer few-shot prompts that combined explicit task instructions with example outputs and the most relevant knowledge graph subsets retrieved via vector embedding similarity, resulting in an 80% improvement in response accuracy.",
    "Built a retrieval-augmented generation (RAG) pipeline that empowered HR professionals to ask high-level, natural language questions against a structured knowledge graph, leveraging OpenAI’s API and Jinja-based prompt engineering to deliver highly accurate responses, improving overall decision-making efficiency.",
    "Leveraged ChatGPT and DeepSeek to help break down development hurdles speeding up implementation and decreasing my use of administrative resources.",
    "Utilized Secure File Transfer Protocol (SFTP) to send files to the AWS instance hosting our web application.",
    "Implemented 50+ REST API endpoints for handling JSON payloads and HTTP requests, reducing CRUD operation time by 30% through dynamic AJAX components and modular design patterns.",
    "Drove full-stack development of a robust HR analytics platform across the full Software Development Life Cycle (SDLC), from requirements gathering and architecture design to testing and deployment, delivering production-ready features in Agile sprints and enhancing HR insights accessibility by 40%.",
    "Built a consumer-facing HR analytics dashboard using JavaScript and Chart.js, translating Neo4j knowledge graph CRUD operations into real-time time-series visualizations. Reduced HR decision latency by 40% and improved anomaly detection by 65% through continuous change monitoring.",
    "Led product research and requirements analysis by evaluating Obsidian’s features and workflows, translating insights into system design and Neo4j schema architecture. This approach enabled the full-stack team to reduce development time by 25% and deliver an internal talent-tracking web app more efficiently.",
    "Engineered retrieval-augmented generation (RAG) pipelines for our HR knowledge graph application, implementing bidirectional relationship extraction and token optimization strategies that improved context relevance by 50% while troubleshooting GPU acceleration to reduce model inference latency by 30%.",
    "Programmed Arduino microcontrollers in C/C++ to design and debug digital circuits (button-controlled LED matrices, sensor-triggered fans), implementing GPIO manipulation and interrupt handling to advance foundational hardware and embedded systems knowledge.",
    "Led Docker containerization and deployment of a consumer-facing HR analytics platform to AWS, collaborating cross-functionally with engineers, cybersecurity specialists, and C-suite stakeholders to improve application availability and enable scalable production rollout.", #= to next 
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
    "Collaborated closely with a team of three Software Engineers to build a full-stack web application using Python and Flask for the backend, and HTML, CSS, JavaScript, and Bootstrap for the frontend, while fostering effective communication that accelerated development and enabled delivery two weeks ahead of schedule.",
    "Developed a flexible data ingestion pipeline in Python using Flask for API integration and Pandas for data processing, enabling quick graph node creation from search bar input or CSV upload via a custom adjacency list syntax, cutting data loading complexity by 70%",
    "Revamped user interfaces with Bootstrap/CSS, implementing modal-based editing and autocomplete search that decreased user errors by 45% in node-relationship management workflows.", 
    "Integrated OpenAI API for question answering, retrieval-augmented generation, leveraged Jinja to perform prompt engineering, which improved response accuracy by 80% while implementing usage caps to control API costs.",
    "Created comprehensive repository wiki documentation for graph schemas and RESTful API endpoints, improving onboarding efficiency by 50% for new developers while standardizing data ingestion protocols.",
    "Developed an elegant graph view using HTML5, CSS, JavaScript, Bootstrap, and Alchemy.js while using AJAX and RESTful API calls for dynamic updates, enabling real-time graph updates that accelerated HR data visualization by 90%.",
    "Implemented schema validation with Python, Flask, and Neo4j, creating node-type checking algorithms that reduced data inconsistencies by 80% in knowledge graph updates.",
    "Refactored 5,000+ lines of Python, HTML, and JavaScript code in an Agile environment using Git, resolving 15+ critical bugs while improving maintainability and reliability by 90%.",
    "Researched Hugging Face models and vector embeddings to utilize in retrieval augmented generation against the data in our knowledge graph application, prototyping local Large Language Model solutions that demonstrated 50% faster query resolution for candidate search scenarios.",
    "Designed responsive user interfaces with Bootstrap/Jinja templates, implementing dynamic layouts by leveraging AJAX and jQuery to modify page layouts without necessitating page refresh, resulting in a more satisfying user experience.",
    "Developed interactive relationship visualizations with Alchemy.js that enabled users to intuitively input and link personnel data, reducing user errors by 25%.",
    "Visualized Human Resource information highlighting skills, certifications, and clearances held by the internal workforce using HTML5, CSS, JavaScript, and Alchemy.js, producing relationship visualization components that decreased user errors by 25% in graph node and relationship creation workflows.",
    "Developed knowledge graph-based search and Natural Language Processing features including autocomplete functionality using Cypher query language, enabling complex relationship mapping across 200+ nodes which improved HR data query efficiency by 60%",
    "Integrated Python/Flask RESTful API endpoints with a JavaScript frontend using AJAX and jQuery, developing dynamic node/relationship management interfaces that reduced CRUD operation time by 30% through real-time form validation and modal-based editing.",
    "Containerized an HR analytics platform using Docker to enable scalable deployment on AWS, streamlining the CI/CD pipeline and reducing deployment errors by 30%.",
    "Documented 20+ RESTful API endpoints and internal code functionality, accelerating onboarding speed by 50% for new software engineering hires.",
    "Created data ingestion systems using Pandas to process CSV data into Neo4j knowledge graphs, streamlining the ingestion of over 200+ nodes and 100+ relationships.",
    "Led frontend bug resolution and feature enhancements in Agile environment using GitHub, refactoring 5K+ lines of front end JavaScript, HTML5, and CSS code into modular components that improved maintainability by 50%.", 
]

#create a client for interacting with the db 
client = chromadb.PersistentClient() 
client.delete_collection("acclaim_work")


collection = client.get_or_create_collection(
    name="acclaim_work", 
   configuration={"hnsw": {"space": "cosine"}}

)

collection.upsert(
    ids=[str(i) for i in range(len(work_experience))],
    documents=work_experience
)

results = collection.query(query_texts="software engineer", n_results=100)

print(results)
print(results["documents"])
NEO4J_URI = "neo4j+s://f82194b9.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your neo4j password"

OPENAI_API_KEY = "your OpenAI API key" 

PINECONE_API_KEY = "your Pinecone API key"
PINECONE_ENV = "us-east1-gcp"   # example
PINECONE_INDEX_NAME = "your pinecone instance/index name"
PINECONE_VECTOR_DIM = 1536       # adjust to embedding model used (text-embedding-3-large ~ 3072? check your model); we assume 1536 for common OpenAI models — change if needed.

REDIS_HOST = "your redis host"
REDIS_PORT = "your redis port"
REDIS_USERNAME = "default"
REDIS_PASSWORD = "your redis password"
REDIS_CACHE_TTL = 86400  #24hours
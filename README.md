# LOL-LM
Satirical News Generator




## RAG

In this project the purpose of RAG is to search for real breaking news, feed that fact to the model, and the model writes a satirical news article about it.

**Without RAG**
The model would just write a satirical news article about a topic that may have happened 3 years ago (depending on the model's training data), or it would just make up a topic that is really far from reality.

This way we are able to improve the model's contextual understanding and reduce hallucinations.


## Dependancies

- LangChain
- Trifalatura (scraping)

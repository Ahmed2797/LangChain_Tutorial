from langchain_classic.text_splitter import (RecursiveCharacterTextSplitter,
                                             Language,CharacterTextSplitter,
                                             )
from langchain_text_splitters import PythonCodeTextSplitter

text = '''
Machine Learning (ML) is a branch of Artificial Intelligence (AI) that allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed.

It is mainly divided into three categories:

Supervised Learning:
The model learns from labeled data.
Example — Predicting house prices using previous house data (price, area, location, etc.).

Unsupervised Learning:
The model finds hidden patterns in unlabeled data.
Example — Customer segmentation based on purchasing behavior.

Reinforcement Learning:
The model learns by interacting with the environment and receiving rewards or penalties.
Example — Teaching a robot to walk or play chess. 
'''

## Language_Spliter
split_text = RecursiveCharacterTextSplitter.from_language(
                                             language=Language.PYTHON,
                                             chunk_size=20,
                                             chunk_overlap=5)




## Markdown_spliter
split_text = RecursiveCharacterTextSplitter.from_language(
                                            language=Language.MARKDOWN,
                                             chunk_size=20,
                                             chunk_overlap=5)




## CharacterTextSplitter
split_text = CharacterTextSplitter(separator='',
                      chunk_size=20,
                      chunk_overlap=5)





chunk = split_text.split_text(text)
print(chunk)

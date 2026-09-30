from sentence_transformer import util, SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
sentence = [
    "I Love playing chess",
    "I enjoy playing football",
    "I like eating Icecream"
]
sentence_embedding = model.encode(sentence)
print(sentence_embedding[0])
similarity1 = util.cos_sim(sentence_embedding[0],sentence_embedding[1])
print(similarity1.item())
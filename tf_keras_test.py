from re import X
import tensorflow as tf
import numpy as np

NUM_ITEMS = 10

ITEM_FEATURE_HASH_SIZE = 100
USER_FEATURE_HASH_SIZE = 1000
EMBEDDING_DIM = 32

class SparseArch(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.user = tf.keras.layers.Embedding(ITEM_FEATURE_HASH_SIZE, EMBEDDING_DIM)
        self.item = tf.keras.layers.Embedding(USER_FEATURE_HASH_SIZE, EMBEDDING_DIM)
        self.dot = tf.keras.layers.Dot(axes=(1))
    
    def call(self, user_features, item_features):
        user_embedding = self.user(user_features)
        item_embeddings = self.item(item_features)
        print("user_embedding: ", user_embedding)
        print("item_embeddings: ", item_embeddings)
        user_embeddings = tf.broadcast_to(user_embedding, shape=(NUM_ITEMS, EMBEDDING_DIM))
        print(user_embeddings)
        print(item_embeddings)
        return tf.keras.activations.sigmoid(self.dot([user_embeddings, item_embeddings]))
model = SparseArch()

x_train_batch = np.array(
    [
    [0, 1, 2, 0, 1, 2], # user feature (e.g., id)
    [0, 0, 1, 0, 0, 1], # item feature (e.g., item country)
    ]
)
y = np.array([12, 23, 87, 23, 14, 95]) # out of NUM_ITEMS (e.g. > 50 country=1)

x_retrieved_candidates = np.array([0 if i < NUM_ITEMS/2 else 1 for i in range(NUM_ITEMS)]) # prediction batch = 4
x_user_accesses = np.array([1,0,1,2])

for user in x_user_accesses:
    user_features = tf.convert_to_tensor(user)
    item_features =tf.convert_to_tensor(x_retrieved_candidates)
    print(model(user_features, item_features))


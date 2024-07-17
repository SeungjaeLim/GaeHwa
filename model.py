import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
import io

# Load LLaVA model
class LLavaModel(torch.nn.Module):
    def __init__(self):
        super(LLavaModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-7b-hf')

    def forward(self, text_inputs, image_path=None):
        input_ids = tokenizer(text_inputs, return_tensors='pt').input_ids
        if image_path:
            image = Image.open(image_path)
            # Process image (dummy implementation)
            image_features = torch.rand(1, 768)  # Replace with actual image processing
            outputs = self.model.generate(input_ids, visual_inputs=image_features)
        else:
            outputs = self.model.generate(input_ids)
        return outputs

# Initialize models and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
llava_model = LLavaModel()

# Example user history and recent product
user_history = ["used_product_1", "used_product_2"]
recent_product = "new_product_description"
user_review = "I liked the last product."
user_photo = "path_to_user_photo.jpg"  # Assuming some image processing is done separately

# Encode user history
def encode_text(model, text_list):
    encoded_inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True)
    outputs = model.generate(**encoded_inputs)
    return outputs

user_history_encoded = encode_text(t5_model, user_history)
recent_product_encoded = encode_text(t5_model, [recent_product])

# Predict sentiment using LLaVA
def predict_sentiment(model, review, product, photo_path):
    input_text = f"Review: {review} Product: {product}"
    outputs = model(input_text, image_path=photo_path)
    sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sentiment

sentiment = predict_sentiment(llava_model, user_review, recent_product, user_photo)
print(f"Predicted sentiment: {sentiment}")

# Use LLaVA to get compatibility vectors
def get_compatibility_vector(model, review, product, photo_path):
    input_text = f"Review: {review} Product: {product}"
    outputs = model(input_text, image_path=photo_path)
    compatibility_vector = outputs[0].detach()
    return compatibility_vector

compatibility_vector = get_compatibility_vector(llava_model, user_review, recent_product, user_photo)

# Recommendation system
def recommend_product(user_vector, candidate_vectors):
    user_vector_np = user_vector.numpy()
    candidate_vectors_np = [vec.numpy() for vec in candidate_vectors]
    similarities = cosine_similarity(user_vector_np, candidate_vectors_np)
    return similarities.argmax()

# Example candidate products (encoded vectors)
candidate_products = [torch.rand(1, 768) for _ in range(10)]
user_vector = compatibility_vector

# Get recommendation
recommended_index = recommend_product(user_vector, candidate_products)
print(f"Recommended product index: {recommended_index}")

# Display recommendation
def display_recommendation(index):
    # Placeholder function to display the recommended product
    print(f"Displaying product {index}")

display_recommendation(recommended_index)

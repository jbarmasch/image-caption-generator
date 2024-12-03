from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import models
from PIL import Image
import models.moondream

# Load the pretrained model and tokenizer
model = models.moondream.MoondreamCaptioner()

# Create FastAPI instance
app = FastAPI()

# Ensure model is in evaluation mode
# model.eval()

# Function to generate caption
def generate_caption(image: Image.Image, temperature, prompt) -> str:
    return model.get_caption(image, temperature=temperature, prompt=prompt)

# Endpoint to receive image and return caption
@app.post("/caption/")
async def create_caption(file: UploadFile = File(...), 
                         temperature: float = Query(None, ge=0.0, description="Optional temperature for caption generation"),
                         prompt: str = Query(None, description="Optional prompt for caption generation")):
    try:
        # Read image file
        image = Image.open(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Generate the caption
    caption = generate_caption(image, temperature=temperature, prompt=prompt)
    
    return {"caption": caption}

# Run the server with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

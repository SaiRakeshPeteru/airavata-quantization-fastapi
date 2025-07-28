from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

app = FastAPI()

# Load model and tokenizer
model_path = "./Quantized_Airavata"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100

@app.post("/generate")
async def generate(req: GenerateRequest):
    result = pipe(req.prompt, max_new_tokens=req.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    return {"generated_text": result[0]["generated_text"]}

@app.get("/")
def root():
    return {"message": "✅ Airavata FastAPI server is running. Visit /docs to try the API."}

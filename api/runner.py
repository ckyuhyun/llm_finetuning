from fastapi import FastAPI
import uvicorn
from llm import LLM_Model

app = FastAPI()



@app.post("/")
async def train_start():
    model = LLM_Model()
    await model.train()
    return {"Result": "Success"}


if __name__ == "__main__":
    uvicorn.run("runner:app", host="127.0.0.1", port=8000)




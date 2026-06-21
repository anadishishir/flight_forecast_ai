# Using A Lightweight Python Base Image 
FROM python:3.12-slim 

# Setting The Working Directory Inside The Container 
WORKDIR /app 

# Copying Requirement File And Install Dependencies 
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt 

# Copying All Project Files Into The Container 
COPY . . 

# Exposing The API Port 
EXPOSE 8000 

# Command To Run FastAPI App 
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]  
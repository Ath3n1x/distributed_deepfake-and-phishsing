## Run Instructions

### Step 1: Create Virtual Environment
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 2: Install Requirements
```
pip install -r requirements.txt
```

### Step 3: Start Webcam App
```
python app/main.py
```

### Step 4: Train the Model
```
python train/train.py --data_dir dataset
```

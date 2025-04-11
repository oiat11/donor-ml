

```
# Donor ML

This project uses a LightGBM-based machine learning model to predict donor behavior (such as attendance or donation likelihood) and write the results back to a MySQL database.

## 📁 Project Structure

```
donor-ml/
├── ml_run_and_update.py     # Main ML script
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables for DB config
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:oiat11/donor-ml.git
cd donor-ml
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate      # On macOS/Linux
# .\venv\Scripts\activate     # On Windows (if needed)
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Configuration

Create a `.env` file in the project root (same level as `ml_run_and_update.py`) with the following content:

```dotenv
DB_USER=your_db_username
DB_PASS=your_db_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=your_db_name
```

Make sure your MySQL database is running and contains the necessary `Donor` table and fields used in the script.

---

## 🚀 Run the ML Script

```bash
python ml_run_and_update.py
```

This will:
- Load donor data from the MySQL database.
- Train LightGBM models for attendance and/or donation prediction.
- Write the prediction scores back into the database.

---


from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import pyarrow.dataset as ds
import pyarrow.compute as pc
from huggingface_hub import HfFileSystem
import os
import time
import json

app = FastAPI(title="1CMR-API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
REPO_ID = "avezattar/1CMR-compact" 
HF_TOKEN = os.getenv("hf_token") 

global_fs = HfFileSystem(token=HF_TOKEN)

# --- File List ---
# NOTE: If you followed the chunking advice from the previous message, 
# your files are now named like 'icmr_part01_chunk000.parquet'.
# Since chunk counts vary, we dynamically list the directory once, 
# but we do it at import time so Vercel caches the result.
# If you haven't chunked them yet, this will just list the 4 big files.
try:
    # We use a broader glob here just in case you chunk them later
    FILE_LIST = global_fs.glob(f"datasets/{REPO_ID}/*.parquet")
    # Filter out the .gitattributes or other non-parquet files just to be safe
    FILE_LIST = [f for f in FILE_LIST if f.endswith('.parquet')]
except Exception as e:
    # Fallback if glob fails (e.g., network issue at cold start)
    print(f"Error globbing files: {e}")
    FILE_LIST = [f"datasets/{REPO_ID}/icmr_part0{i}.parquet" for i in range(1, 5)]

@app.get("/")
def root():
    return {"status": "online", "message": "1CMR API is Live."}

@app.get("/search")
def search(mobile: str = Query(..., description="10-digit mobile number")):
    start_time = time.time()
    
    try:
        if len(mobile) != 10 or not mobile.isdigit():
            return Response(
                content=json.dumps({"error": "Please provide a valid 10-digit mobile number."}, indent=4), 
                media_type="application/json"
            )

        if not FILE_LIST:
            return Response(
                content=json.dumps({"error": "Dataset files not found or failed to load."}, indent=4), 
                media_type="application/json"
            )

        # Load the dataset
        dataset = ds.dataset(FILE_LIST, filesystem=global_fs, format="parquet")
        
        # Only read the columns we actually need for the response to save memory
        columns_to_read = [
            "name", "fathersName", "phoneNumber", "otherNumber",
            "passportNumber", "aadharNumber", "age", "gender",
            "address", "district", "pincode", "state", "town"
        ]
        
        filter_expr = pc.equal(pc.field("phoneNumber"), mobile)
        
        # use_threads=False prevents Vercel 1GB RAM OOM crash
        # pre_buffer=False reduces memory overhead during read
        table = dataset.scanner(
            filter=filter_expr,
            columns=columns_to_read,
            use_threads=False,
            pre_buffer=False
        ).head(5)
        
        raw_records = table.to_pylist()
        
        elapsed = time.time() - start_time
        
        response_data = {
            "meta": {
                "credits": "Made by Mr. Anonymous",
                "search_query": mobile,
                "total_records": len(raw_records),
                "execution_time": f"{elapsed:.3f}s"
            },
            "results": []
        }

        if not raw_records:
            response_data["message"] = "No record found in the database."
            return Response(
                content=json.dumps(response_data, indent=4), 
                media_type="application/json"
            )
            
        for i, row in enumerate(raw_records, 1):
            # Mapping every column exactly as it appears in the 1CMR screenshots
            clean_row = {}
            if row.get("name"): clean_row["name"] = row["name"]
            if row.get("fathersName"): clean_row["father_name"] = row["fathersName"]
            if row.get("phoneNumber"): clean_row["mobile"] = row["phoneNumber"]
            if row.get("otherNumber"): clean_row["alternate_mobile"] = row["otherNumber"]
            if row.get("passportNumber"): clean_row["passport_number"] = row["passportNumber"]
            if row.get("aadharNumber"): clean_row["aadhar_number"] = row["aadharNumber"]
            if row.get("age"): clean_row["age"] = row["age"]
            if row.get("gender"): clean_row["gender"] = row["gender"]
            if row.get("address"): clean_row["address"] = row["address"]
            if row.get("district"): clean_row["district"] = row["district"]
            if row.get("pincode"): clean_row["pincode"] = row["pincode"]
            if row.get("state"): clean_row["state"] = row["state"]
            if row.get("town"): clean_row["town"] = row["town"]
            
            clean_row["result_id"] = i
            response_data["results"].append(clean_row)
            
        return Response(
            content=json.dumps(response_data, indent=4, default=str), 
            media_type="application/json"
        )
        
    except Exception as e:
        return Response(
            content=json.dumps({"error": str(e)}, indent=4), 
            media_type="application/json"
        )
